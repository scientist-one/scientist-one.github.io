import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class INatSubset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join("/data", row['file_name'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
        
        return self.transform(image), row['category_id']

def compute_visual_prototypes():
    with open("/data/train2019.json", "r") as f:
        data = json.load(f)
    
    # Load train split
    train_ids_df = pd.read_csv("metadata/train_ids.csv")
    train_ids = set(train_ids_df['image_id'].values)
    
    annotations = [ann for ann in data['annotations'] if ann['image_id'] in train_ids]
    images_dict = {img['id']: img['file_name'] for img in data['images']}
    
    records = []
    for ann in annotations:
        records.append({
            'image_id': ann['image_id'],
            'category_id': ann['category_id'],
            'file_name': images_dict[ann['image_id']]
        })
    df = pd.DataFrame(records)
    
    # Sample up to 50 images per class
    sampled_df = df.groupby('category_id').head(50).reset_index(drop=True)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = INatSubset(sampled_df, transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    
    # Model
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    
    class MultiStageModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.features = base_model.features
            
        def forward(self, x):
            outputs = []
            # features is a sequential of 8 blocks
            # 0: stem (Conv2d, LayerNorm)
            # 1: stage 1 (Sequential)
            # 2: downsample (LayerNorm, Conv2d)
            # 3: stage 2
            # 4: downsample
            # 5: stage 3
            # 6: downsample
            # 7: stage 4
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in [1, 3, 5, 7]: # end of stages
                    # global average pool
                    pooled = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
                    outputs.append(pooled)
            return outputs
            
    feature_extractor = MultiStageModel(model).cuda().eval()
    
    num_classes = 1010
    sum_features = {stage: torch.zeros(num_classes, dim).cuda() for stage, dim in enumerate([96, 192, 384, 768])}
    counts = torch.zeros(num_classes).cuda()
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.cuda()
            labels = labels.cuda()
            
            with torch.amp.autocast('cuda'):
                feats = feature_extractor(images)
            
            for stage, feat in enumerate(feats):
                sum_features[stage].index_add_(0, labels, feat.float())
            counts.index_add_(0, labels, torch.ones_like(labels).float())
            
    prototypes = []
    for stage in range(4):
        # average
        mean_feat = sum_features[stage] / counts.unsqueeze(1).clamp(min=1)
        # l2 normalize
        mean_feat = F.normalize(mean_feat, p=2, dim=1)
        prototypes.append(mean_feat.cpu().numpy())
        
    return prototypes

if __name__ == "__main__":
    import torch.nn.functional as F
    prototypes = compute_visual_prototypes()
    np.savez("visual_prototypes.npz", p0=prototypes[0], p1=prototypes[1], p2=prototypes[2], p3=prototypes[3])
    
    # Compute visual similarity matrix V
    V = np.zeros((1010, 1010))
    for p in prototypes:
        sim = p @ p.T # cosine similarity
        V += np.clip(sim, 0, 1) # Only positive similarities
    V /= len(prototypes)
    
    np.save("visual_sim.npy", V)
    print("V shape:", V.shape)
    print("Mean V:", V.mean())
