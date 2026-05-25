import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

NUM_CLASSES = 1010
EPOCHS = 7
BATCH_SIZE = 128
LR = 5e-4
WEIGHT_DECAY = 1e-4

class INatDataset(Dataset):
    def __init__(self, df, transform, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join("/data", row['file_name'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
            
        tensor = self.transform(image)
        if self.is_test:
            return tensor, row['id']
        else:
            return tensor, row['category_id']

def compute_taxonomic_similarity(data):
    categories = sorted(data['categories'], key=lambda x: x['id'])
    T = np.zeros((NUM_CLASSES, NUM_CLASSES))
    levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i == j:
                T[i, j] = 1.0
                continue
            shared = sum(1 for lvl in levels if categories[i][lvl] == categories[j][lvl])
            T[i, j] = shared / len(levels)
    return T

def compute_visual_prototypes(df, data):
    sampled_df = df.groupby('category_id').head(30).reset_index(drop=True)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = INatDataset(sampled_df, transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    
    class MultiStageModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.features = base_model.features
            
        def forward(self, x):
            outputs = []
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in [1, 3, 5, 7]:
                    outputs.append(F.adaptive_avg_pool2d(x, (1, 1)).flatten(1))
            return outputs
            
    extractor = MultiStageModel(model).cuda().eval()
    sum_features = {stage: torch.zeros(NUM_CLASSES, dim).cuda() for stage, dim in enumerate([96, 192, 384, 768])}
    counts = torch.zeros(NUM_CLASSES).cuda()
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            with torch.amp.autocast('cuda'):
                feats = extractor(images)
            for stage, feat in enumerate(feats):
                sum_features[stage].index_add_(0, labels, feat.float())
            counts.index_add_(0, labels, torch.ones_like(labels).float())
            
    V = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for stage in range(4):
        mean_feat = sum_features[stage] / counts.unsqueeze(1).clamp(min=1)
        mean_feat = F.normalize(mean_feat, p=2, dim=1).cpu().numpy()
        sim = mean_feat @ mean_feat.T
        V += np.clip(sim, 0, 1)
    V /= 4
    return V

def main():
    # Load metadata
    with open("/data/train2019.json", "r") as f:
        data = json.load(f)
        
    train_ids_df = pd.read_csv("/workspace/metadata/train_ids.csv")
    val_ids_df = pd.read_csv("/workspace/metadata/val_ids.csv")
    
    train_ids = set(train_ids_df['image_id'].values)
    val_ids = set(val_ids_df['image_id'].values)
    
    images_dict = {img['id']: img['file_name'] for img in data['images']}
    
    train_records, val_records = [], []
    for ann in data['annotations']:
        rec = {'image_id': ann['image_id'], 'category_id': ann['category_id'], 'file_name': images_dict[ann['image_id']]}
        if ann['image_id'] in train_ids:
            train_records.append(rec)
        elif ann['image_id'] in val_ids:
            val_records.append(rec)
            
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    
    print("Computing similarities...")
    T = compute_taxonomic_similarity(data)
    V = compute_visual_prototypes(train_df, data)
    
    V_min, V_max = V.min(), V.max()
    V_norm = (V - V_min) / (V_max - V_min + 1e-8)
    np.fill_diagonal(V_norm, 1.0)
    
    # Multiplicative fusion to filter out spurious visual similarity
    # between taxonomically distant classes
    S = V_norm * T
    
    beta = 1.0
    M = beta * S
    np.fill_diagonal(M, 0)
    M_tensor = torch.tensor(M, dtype=torch.float32).cuda()
    print("Similarities computed.")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = INatDataset(train_df, train_transform)
    val_dataset = INatDataset(val_df, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    model = model.cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(images)
                # Apply pairwise margin
                margins = M_tensor[labels]
                adjusted_logits = logits + margins
                loss = F.cross_entropy(adjusted_logits, labels, label_smoothing=0.1)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            old_scale = scaler.get_scale()
            scaler.update()
            new_scale = scaler.get_scale()
            if new_scale >= old_scale:
                scheduler.step()
            
            train_loss += loss.item()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_acc = correct / total
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            
    print(f"VAL_METRIC: {1.0 - best_val_acc:.4f}")
    
    # Inference
    with open("/data/test2019.json", "r") as f:
        test_data = json.load(f)
    test_df = pd.DataFrame(test_data['images'])
    
    test_dataset = INatDataset(test_df, val_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    preds = []
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Testing"):
            images = images.cuda()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            top5 = outputs.topk(5, 1)[1].cpu().numpy()
            for i, img_id in enumerate(ids.numpy()):
                preds.append({'id': img_id, 'predicted': " ".join(map(str, top5[i]))})
                
    sub_df = pd.DataFrame(preds)
    sub_df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
