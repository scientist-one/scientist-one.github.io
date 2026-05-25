import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from bisect import bisect
import pathlib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions

def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0
    for gt, pred in zip(ground_truth, predictions):
        ranks = {cell_id: i for i, cell_id in enumerate(gt)}
        pred_ranks = [ranks[x] for x in pred if x in ranks]
        n = len(gt)
        total_2max += n * (n - 1)
        total_inversions += count_inversions(pred_ranks)
    return 1 - 4 * total_inversions / total_2max

class DualEncoder(nn.Module):
    def __init__(self, model_name='microsoft/codebert-base'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, md_input_ids, md_attention_mask, bnd_input_ids, bnd_attention_mask):
        md_feat = self.encoder(input_ids=md_input_ids, attention_mask=md_attention_mask)[0][:, 0]
        bnd_feat = self.encoder(input_ids=bnd_input_ids, attention_mask=bnd_attention_mask)[0][:, 0]
        
        md_feat = F.normalize(md_feat, p=2, dim=-1)
        bnd_feat = F.normalize(bnd_feat, p=2, dim=-1)
        
        return md_feat, bnd_feat

    def encode_text(self, input_ids, attention_mask):
        feat = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        return F.normalize(feat, p=2, dim=-1)

def get_boundaries(nb_id, cell_types, source, max_len=512):
    code_cells = [c for c, t in cell_types.items() if t == 'code']
    code_count = len(code_cells)
    
    boundaries = []
    # boundary 0: before first code cell
    # boundary k: between code cell k-1 and k
    # boundary C: after last code cell
    for k in range(code_count + 1):
        prev_code = source[code_cells[k-1]] if k > 0 else ""
        
        # truncate to max_len chars to save tokenization time
        prev_code = prev_code[-max_len:] 
        
        bnd_text = f"Previous: {prev_code}"
        boundaries.append(bnd_text)
    return code_cells, boundaries

def prepare_training_pairs(ids, orders_df):
    orders_dict = dict(zip(orders_df['id'], orders_df['cell_order']))
    md_texts = []
    bnd_texts = []
    
    for nb_id in tqdm(ids, desc="Preparing pairs"):
        with open(f'/data/train/{nb_id}.json', 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        cell_types = nb['cell_type']
        source = nb['source']
        order = orders_dict[nb_id].split()
        
        code_cells, boundaries = get_boundaries(nb_id, cell_types, source)
        
        current_k = 0
        for cell_id in order:
            if cell_types[cell_id] == 'code':
                current_k += 1
            else:
                md_texts.append(source[cell_id])
                bnd_texts.append(boundaries[current_k])
                
    return md_texts, bnd_texts

class NBDataset(Dataset):
    def __init__(self, md_texts, bnd_texts, tokenizer, max_len=128):
        self.md_texts = md_texts
        self.bnd_texts = bnd_texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.md_texts)
        
    def __getitem__(self, idx):
        md = self.tokenizer(self.md_texts[idx], max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        bnd = self.tokenizer(self.bnd_texts[idx], max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        return {
            'md_input_ids': md['input_ids'].squeeze(0),
            'md_attention_mask': md['attention_mask'].squeeze(0),
            'bnd_input_ids': bnd['input_ids'].squeeze(0),
            'bnd_attention_mask': bnd['attention_mask'].squeeze(0)
        }

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'microsoft/codebert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DualEncoder(model_name).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    train_ids = pd.read_csv('/workspace/metadata/train_ids.csv')['id'].values
    orders_df = pd.read_csv('/data/train_orders.csv')
    
    # Sample 50000 notebooks for quick training
    np.random.seed(42)
    train_ids = np.random.choice(train_ids, 30000, replace=False)
    
    md_texts, bnd_texts = prepare_training_pairs(train_ids, orders_df)
    
    dataset = NBDataset(md_texts, bnd_texts, tokenizer, max_len=256)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    margin_loss_fn = nn.MarginRankingLoss(margin=0.1)
    
    model.train()
    epochs = 2
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            
            md_input_ids = batch['md_input_ids'].to(device)
            md_attention_mask = batch['md_attention_mask'].to(device)
            bnd_input_ids = batch['bnd_input_ids'].to(device)
            bnd_attention_mask = batch['bnd_attention_mask'].to(device)
            
            if isinstance(model, nn.DataParallel):
                md_feat, bnd_feat = model(md_input_ids, md_attention_mask, bnd_input_ids, bnd_attention_mask)
                scale = model.module.logit_scale.exp()
            else:
                md_feat, bnd_feat = model(md_input_ids, md_attention_mask, bnd_input_ids, bnd_attention_mask)
                scale = model.logit_scale.exp()
                
            logits = scale * md_feat @ bnd_feat.t()
            labels = torch.arange(logits.size(0)).to(device)
            loss_ce = (loss_fn(logits, labels) + loss_fn(logits.t(), labels)) / 2
            
            # Ranking loss: correct boundary should have higher score than other boundaries
            # Diagonal holds correct similarities
            pos_sim = torch.diag(logits) # [B]
            
            # Off-diagonal (negative boundaries)
            # Sample negative indices: shift by 1 or random
            neg_idx = (labels + 1) % logits.size(0)
            neg_sim = logits[labels, neg_idx]
            
            target = torch.ones_like(pos_sim)
            loss_margin = margin_loss_fn(pos_sim, neg_sim, target)
            
            loss = loss_ce + 0.5 * loss_margin
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item()})
            
    # Save model weights
    out_dir = pathlib.Path(__file__).resolve().parent
    torch.save(model.state_dict(), out_dir / 'dual_encoder.pth')
    
    return tokenizer, model

def validate_and_predict(tokenizer, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    val_ids = pd.read_csv('/workspace/metadata/val_ids.csv')['id'].values
    orders_df = pd.read_csv('/data/train_orders.csv')
    orders_dict = dict(zip(orders_df['id'], orders_df['cell_order']))
    
    val_preds = []
    val_gts = []
    
    # Use continuous position expectation
    with torch.no_grad():
        for nb_id in tqdm(val_ids, desc="Validating"):
            with open(f'/data/train/{nb_id}.json', 'r', encoding='utf-8') as f:
                nb = json.load(f)
                
            cell_types = nb['cell_type']
            source = nb['source']
            code_cells, boundaries = get_boundaries(nb_id, cell_types, source)
            md_cells = [c for c, t in cell_types.items() if t == 'markdown']
            
            if not md_cells:
                val_preds.append(code_cells)
                val_gts.append(orders_dict[nb_id].split())
                continue
                
            # Encode boundaries
            bnd_encs = []
            for i in range(0, len(boundaries), 64):
                batch_texts = boundaries[i:i+64]
                t = tokenizer(batch_texts, max_length=256, padding=True, truncation=True, return_tensors='pt').to(device)
                if isinstance(model, nn.DataParallel):
                    emb = model.module.encode_text(t['input_ids'], t['attention_mask'])
                else:
                    emb = model.encode_text(t['input_ids'], t['attention_mask'])
                bnd_encs.append(emb)
            bnd_embs = torch.cat(bnd_encs, dim=0) # [C+1, D]
            
            # Encode mds
            md_encs = []
            md_texts = [source[m_id] for m_id in md_cells]
            for i in range(0, len(md_texts), 64):
                batch_texts = md_texts[i:i+64]
                t = tokenizer(batch_texts, max_length=256, padding=True, truncation=True, return_tensors='pt').to(device)
                if isinstance(model, nn.DataParallel):
                    emb = model.module.encode_text(t['input_ids'], t['attention_mask'])
                else:
                    emb = model.encode_text(t['input_ids'], t['attention_mask'])
                md_encs.append(emb)
            md_embs = torch.cat(md_encs, dim=0) # [M, D]
            
            # Compute similarities
            if isinstance(model, nn.DataParallel):
                scale = model.module.logit_scale.exp()
            else:
                scale = model.logit_scale.exp()
                
            sims = scale * md_embs @ bnd_embs.t() # [M, C+1]
            probs = F.softmax(sims, dim=1) # [M, C+1]
            
            # Expected boundary position
            positions = torch.arange(len(boundaries)).to(device, dtype=torch.float32)
            exp_pos = (probs * positions).sum(dim=1).cpu().numpy()
            
            # Assign ranks
            cell_ranks = {}
            for i, c in enumerate(code_cells):
                cell_ranks[c] = i + 0.5
                
            for i, c in enumerate(md_cells):
                cell_ranks[c] = exp_pos[i]
                
            pred_order = sorted(cell_ranks.keys(), key=lambda x: cell_ranks[x])
            val_preds.append(pred_order)
            val_gts.append(orders_dict[nb_id].split())
            
    score = kendall_tau(val_gts, val_preds)
    print(f"VAL_METRIC: {score}")
    
    # Predict Test
    test_ids = [f.split('.')[0] for f in os.listdir('/data/test') if f.endswith('.json')]
    sub_rows = []
    
    with torch.no_grad():
        for nb_id in tqdm(test_ids, desc="Predicting Test"):
            with open(f'/data/test/{nb_id}.json', 'r', encoding='utf-8') as f:
                nb = json.load(f)
                
            cell_types = nb['cell_type']
            source = nb['source']
            code_cells, boundaries = get_boundaries(nb_id, cell_types, source)
            md_cells = [c for c, t in cell_types.items() if t == 'markdown']
            
            if not md_cells:
                sub_rows.append({'id': nb_id, 'cell_order': ' '.join(code_cells)})
                continue
                
            bnd_encs = []
            for i in range(0, len(boundaries), 64):
                batch_texts = boundaries[i:i+64]
                t = tokenizer(batch_texts, max_length=256, padding=True, truncation=True, return_tensors='pt').to(device)
                if isinstance(model, nn.DataParallel):
                    emb = model.module.encode_text(t['input_ids'], t['attention_mask'])
                else:
                    emb = model.encode_text(t['input_ids'], t['attention_mask'])
                bnd_encs.append(emb)
            bnd_embs = torch.cat(bnd_encs, dim=0) # [C+1, D]
            
            md_encs = []
            md_texts = [source[m_id] for m_id in md_cells]
            for i in range(0, len(md_texts), 64):
                batch_texts = md_texts[i:i+64]
                t = tokenizer(batch_texts, max_length=256, padding=True, truncation=True, return_tensors='pt').to(device)
                if isinstance(model, nn.DataParallel):
                    emb = model.module.encode_text(t['input_ids'], t['attention_mask'])
                else:
                    emb = model.encode_text(t['input_ids'], t['attention_mask'])
                md_encs.append(emb)
            md_embs = torch.cat(md_encs, dim=0) # [M, D]
            
            if isinstance(model, nn.DataParallel):
                scale = model.module.logit_scale.exp()
            else:
                scale = model.logit_scale.exp()
                
            sims = scale * md_embs @ bnd_embs.t()
            probs = F.softmax(sims, dim=1)
            
            positions = torch.arange(len(boundaries)).to(device, dtype=torch.float32)
            exp_pos = (probs * positions).sum(dim=1).cpu().numpy()
            
            cell_ranks = {}
            for i, c in enumerate(code_cells):
                cell_ranks[c] = i + 0.5
            for i, c in enumerate(md_cells):
                cell_ranks[c] = exp_pos[i]
                
            pred_order = sorted(cell_ranks.keys(), key=lambda x: cell_ranks[x])
            sub_rows.append({'id': nb_id, 'cell_order': ' '.join(pred_order)})
            
    sub_df = pd.DataFrame(sub_rows)
    out_dir = pathlib.Path(__file__).resolve().parent
    sub_df.to_csv(out_dir / 'submission.csv', index=False)
    print("Saved submission.csv")

if __name__ == '__main__':
    tokenizer, model = train()
    validate_and_predict(tokenizer, model)
