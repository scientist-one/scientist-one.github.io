import os
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 3474
DINO_MODEL_NAME = 'vit_giant_patch14_reg4_dinov2.lvd142m'
CACHE_DIR = "/workspace/.cache/my_features"

class ImgDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
            
        target = np.zeros(NUM_CLASSES, dtype=np.float32)
        if 'attribute_ids' in row and pd.notna(row['attribute_ids']):
            labels = [int(x) for x in str(row['attribute_ids']).split()]
            target[labels] = 1.0
            
        return img, target, img_id

class DINOv2Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(DINO_MODEL_NAME, pretrained=True, num_classes=0, dynamic_img_size=True)
    def forward(self, x):
        out = self.model.forward_features(x)
        cls_token = out[:, 0]
        patch_tokens = out[:, self.model.num_prefix_tokens:].mean(dim=1)
        return torch.cat([cls_token, patch_tokens], dim=1)

def extract_features(df, img_dir, split_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    features_path = os.path.join(CACHE_DIR, f"{split_name}_features_336_v2.pt")
    targets_path = os.path.join(CACHE_DIR, f"{split_name}_targets_336_v2.pt")
    ids_path = os.path.join(CACHE_DIR, f"{split_name}_ids_336_v2.csv")
    
    if os.path.exists(features_path):
        print(f"Loading cached features from {features_path}")
        return torch.load(features_path), torch.load(targets_path), pd.read_csv(ids_path)
        
    print(f"Extracting features for {split_name}...")
    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    dataset = ImgDataset(df, img_dir, transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    extractor = DINOv2Extractor().to(DEVICE).eval()
    
    all_features = []
    all_targets = []
    all_ids = []
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for imgs, targets, ids in tqdm(loader, desc=f"Extracting {split_name}"):
                imgs = imgs.to(DEVICE, non_blocking=True)
                feats = extractor(imgs)
                all_features.append(feats.cpu())
                all_targets.append(targets)
                all_ids.extend(ids)
                
    all_features = torch.cat(all_features, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    ids_df = pd.DataFrame({'id': all_ids})
    
    torch.save(all_features, features_path)
    torch.save(all_targets, targets_path)
    ids_df.to_csv(ids_path, index=False)
    
    del extractor
    torch.cuda.empty_cache()
    gc.collect()
    
    return all_features, all_targets, ids_df

def compute_hierarchical_pairs(train_targets, threshold=0.95):
    occurrences = train_targets.sum(dim=0)
    valid_classes = (occurrences > 10).float()
    
    co_occurrences = torch.matmul(train_targets.T, train_targets)
    prob_A_given_B = co_occurrences / (occurrences.unsqueeze(0) + 1e-8)
    prob_A_given_B.fill_diagonal_(0)
    prob_A_given_B = prob_A_given_B * valid_classes.unsqueeze(0) * valid_classes.unsqueeze(1)
    
    indices = torch.where(prob_A_given_B > threshold)
    a_idx = indices[0].cpu().numpy().tolist()
    b_idx = indices[1].cpu().numpy().tolist()
    weights = prob_A_given_B[indices].cpu().numpy().tolist()
    
    print(f"Found {len(a_idx)} hierarchical pairs with P(A|B) > {threshold}")
    return a_idx, b_idx, weights

class AsymmetricHierarchicalLoss(nn.Module):
    def __init__(self, a_idx=[], b_idx=[], weights=[], gamma_neg=4.0, gamma_pos=1.0, clip=0.05, lambda_hier=0.1, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.lambda_hier = lambda_hier
        
        self.a_idx = torch.tensor(a_idx, dtype=torch.long).to(DEVICE)
        self.b_idx = torch.tensor(b_idx, dtype=torch.long).to(DEVICE)
        self.weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        
    def forward(self, logits, targets):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid
        
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
            
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss = loss * one_sided_w
            
        base_loss = -loss.sum() / logits.size(0)
        
        if self.lambda_hier > 0 and len(self.a_idx) > 0:
            p_A = x_sigmoid[:, self.a_idx]
            p_B = x_sigmoid[:, self.b_idx]
            diff = F.relu(p_B - p_A)
            hier_loss = (self.weights * (diff ** 2)).sum() / logits.size(0)
            return base_loss + self.lambda_hier * hier_loss
            
        return base_loss

class MLP(nn.Module):
    def __init__(self, input_dim=3072, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class FeatureDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def train_ensemble(train_features, train_targets, val_features, val_targets, a_idx, b_idx, weights):
    num_models = 10
    epochs = 120
    batch_size = 1024
    
    train_dataset = FeatureDataset(train_features, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    val_preds = np.zeros((num_models, len(val_features), NUM_CLASSES))
    
    for m in range(num_models):
        print(f"Training Model {m+1}/{num_models}")
        model = MLP().to(DEVICE)
        criterion = AsymmetricHierarchicalLoss(a_idx, b_idx, weights, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, lambda_hier=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(val_features), 1024):
                x = val_features[i:i+1024].to(DEVICE)
                preds.append(torch.sigmoid(model(x)).cpu().numpy())
            val_preds[m] = np.concatenate(preds)
            
        torch.save(model.state_dict(), f"model_{m}.pt")
        
    return val_preds

def find_best_threshold(val_preds, val_targets):
    best_f1 = 0
    best_thresh = 0.1
    for t in np.arange(0.1, 0.7, 0.01):
        preds_t = (val_preds > t).astype(np.float32)
        tp = (preds_t * val_targets).sum()
        fp = (preds_t * (1 - val_targets)).sum()
        fn = ((1 - preds_t) * val_targets).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    return best_f1, best_thresh

def main():
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
    os.environ['TORCH_HOME'] = '/workspace/.cache/torch'
    
    train_df = pd.read_csv('/data/train.csv')
    train_ids = pd.read_csv('/workspace/metadata/train_ids.csv')['id'].values
    val_ids = pd.read_csv('/workspace/metadata/val_ids.csv')['id'].values
    
    train_split = train_df[train_df['id'].isin(train_ids)].copy()
    val_split = train_df[train_df['id'].isin(val_ids)].copy()
    
    train_features, train_targets, _ = extract_features(train_split, '/data/train', 'train')
    val_features, val_targets, _ = extract_features(val_split, '/data/train', 'val')
    
    a_idx, b_idx, weights = compute_hierarchical_pairs(train_targets, threshold=0.95)
    
    val_preds_models = train_ensemble(train_features, train_targets, val_features, val_targets, a_idx, b_idx, weights)
    val_preds_ensemble = val_preds_models.mean(axis=0)
    
    val_targets_np = val_targets.numpy()
    best_f1, best_thresh = find_best_threshold(val_preds_ensemble, val_targets_np)
    
    print(f"VAL_METRIC: {best_f1}")
    print(f"Best threshold: {best_thresh}")
    
    test_df = pd.read_csv('/data/sample_submission.csv')
    test_features, _, test_ids_df = extract_features(test_df, '/data/test', 'test')
    
    test_preds_models = np.zeros((10, len(test_features), NUM_CLASSES))
    for m in range(10):
        model = MLP().to(DEVICE)
        model.load_state_dict(torch.load(f"model_{m}.pt"))
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(test_features), 1024):
                x = test_features[i:i+1024].to(DEVICE)
                preds.append(torch.sigmoid(model(x)).cpu().numpy())
            test_preds_models[m] = np.concatenate(preds)
            
    test_preds_ensemble = test_preds_models.mean(axis=0)
    test_preds_bin = (test_preds_ensemble > best_thresh).astype(int)
    
    submission = []
    for i, img_id in enumerate(test_ids_df['id']):
        labels = np.where(test_preds_bin[i] == 1)[0]
        labels_str = " ".join([str(l) for l in labels])
        submission.append({'id': img_id, 'attribute_ids': labels_str})
        
    sub_df = pd.DataFrame(submission)
    
    # ensure format matches sample submission
    sample_sub = pd.read_csv('/data/sample_submission.csv')
    sub_df = sample_sub[['id']].merge(sub_df, on='id', how='left')
    sub_df['attribute_ids'] = sub_df['attribute_ids'].fillna('')
    
    sub_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
