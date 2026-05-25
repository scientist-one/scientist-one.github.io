import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from pyquaternion import Quaternion
from pathlib import Path
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
import math

import time
import gc

# ----------------- Configuration -----------------
RANDOM_STATE = 42
BEV_RES = 0.2
BEV_X_MIN, BEV_X_MAX = -51.2, 51.2
BEV_Y_MIN, BEV_Y_MAX = -51.2, 51.2
BEV_Z_MIN, BEV_Z_MAX = -5.0, 3.0
Z_SLICES = 8
Z_RES = (BEV_Z_MAX - BEV_Z_MIN) / Z_SLICES
BEV_H = int((BEV_Y_MAX - BEV_Y_MIN) / BEV_RES)
BEV_W = int((BEV_X_MAX - BEV_X_MIN) / BEV_RES)
STRIDE = 4
FM_H = BEV_H // STRIDE
FM_W = BEV_W // STRIDE

CLASSES = [
    'car', 'motorcycle', 'bus', 'bicycle', 'truck', 
    'pedestrian', 'other_vehicle', 'animal', 'emergency_vehicle'
]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

EPOCHS = 2
BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_HSC_PE = False # Toggle for HSC-PE
# -------------------------------------------------

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

class LyftBEVDataset(Dataset):
    def __init__(self, lyft_data, sample_tokens, is_train=True):
        self.lyft_data = lyft_data
        self.sample_tokens = sample_tokens
        self.is_train = is_train

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        token = self.sample_tokens[idx]
        sample = self.lyft_data.get('sample', token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.lyft_data.get('sample_data', lidar_token)
        lidar_filepath = self.lyft_data.get_sample_data_path(lidar_token)
        
        # Load Point Cloud
        try:
            pc = LidarPointCloud.from_file(lidar_filepath)
        except ValueError:
            # Fallback for corrupted lidar files
            pts = np.zeros((3, 0), dtype=np.float32)
        else:
            pts = pc.points[:3, :] # x, y, z
        
        # Sensor to Ego
        cs_record = self.lyft_data.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        sensor_rot = Quaternion(cs_record['rotation'])
        sensor_trans = np.array(cs_record['translation'])
        pts = sensor_rot.rotation_matrix @ pts + sensor_trans[:, None]
        
        # Ego to Global (Optional, let's keep it in ego frame for BEV)
        # Actually it's better to keep point cloud in Ego Frame for consistent learning.
        
        # Crop points
        mask_x = (pts[0] >= BEV_X_MIN) & (pts[0] < BEV_X_MAX)
        mask_y = (pts[1] >= BEV_Y_MIN) & (pts[1] < BEV_Y_MAX)
        mask_z = (pts[2] >= BEV_Z_MIN) & (pts[2] < BEV_Z_MAX)
        valid_mask = mask_x & mask_y & mask_z
        pts = pts[:, valid_mask]
        
        # Voxelize
        x_idx = ((pts[0] - BEV_X_MIN) / BEV_RES).astype(np.int32)
        y_idx = ((pts[1] - BEV_Y_MIN) / BEV_RES).astype(np.int32)
        z_idx = ((pts[2] - BEV_Z_MIN) / Z_RES).astype(np.int32)
        z_idx = np.clip(z_idx, 0, Z_SLICES - 1)
        
        bev_img = np.zeros((Z_SLICES, BEV_H, BEV_W), dtype=np.float32)
        np.add.at(bev_img, (z_idx, y_idx, x_idx), 1)
        bev_img = np.log1p(bev_img) # log density
        
        # Semantic Map Masks
        # In this dataset, map masks are available.
        # But for the baseline, to avoid the complexity of map masking and coordinate transforms,
        # let's just use the 8-channel lidar BEV. Map fusion will be done in the next solution.
        input_tensor = bev_img # 8 x H x W
        
        if not self.is_train:
            return input_tensor, token
            
        # Targets
        heatmap = np.zeros((NUM_CLASSES, FM_H, FM_W), dtype=np.float32)
        reg_target = np.zeros((8, FM_H, FM_W), dtype=np.float32)
        reg_mask = np.zeros((1, FM_H, FM_W), dtype=np.float32)
        
        _, boxes, _ = self.lyft_data.get_sample_data(lidar_token) # Returns boxes in global frame
        
        # We need boxes in ego frame
        ep_record = self.lyft_data.get('ego_pose', lidar_data['ego_pose_token'])
        ego_rot = Quaternion(ep_record['rotation'])
        ego_trans = np.array(ep_record['translation'])
        
        for box in boxes:
            box.rotate(sensor_rot)
            box.translate(sensor_trans)
            
            cls_name = box.name
            if cls_name not in CLASS2IDX:
                continue
            cls_idx = CLASS2IDX[cls_name]
            
            x, y, z = box.center
            
            x_img = (x - BEV_X_MIN) / BEV_RES
            y_img = (y - BEV_Y_MIN) / BEV_RES
            
            xf = x_img / STRIDE
            yf = y_img / STRIDE
            cx, cy = int(xf), int(yf)
            
            if 0 <= cx < FM_W and 0 <= cy < FM_H:
                w, l, h = box.wlh
                w = max(w, 1e-4)
                l = max(l, 1e-4)
                h = max(h, 1e-4)
                # Yaw in ego frame
                yaw = box.orientation.yaw_pitch_roll[0]
                
                radius = max(2, int((w / BEV_RES) / STRIDE / 2))
                draw_umich_gaussian(heatmap[cls_idx], (cx, cy), radius)
                
                reg_target[:, cy, cx] = [
                    xf - cx, yf - cy, z,
                    np.log(w), np.log(l), np.log(h),
                    np.sin(yaw), np.cos(yaw)
                ]
                reg_mask[0, cy, cx] = 1.0
                
        return input_tensor, heatmap, reg_target, reg_mask

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=8, num_classes=9):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Deconv FPN
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.head_heatmap = nn.Conv2d(64, num_classes, 1)
        self.head_reg = nn.Conv2d(64, 8, 1)
        
        nn.init.constant_(self.head_heatmap.bias, -2.19)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        
        heatmap = torch.sigmoid(self.head_heatmap(x))
        reg = self.head_reg(x)
        return heatmap, reg

def focal_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    pred = torch.clamp(pred, 1e-4, 1 - 1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos

def decode_predictions(heatmaps, reg_preds, K=50):
    batch_size = heatmaps.shape[0]
    
    # heatmaps: B x C x H x W
    # NMS via max pooling
    hmax = nn.functional.max_pool2d(heatmaps, kernel_size=3, stride=1, padding=1)
    keep = (hmax == heatmaps).float()
    heatmaps = heatmaps * keep
    
    heatmaps = heatmaps.view(batch_size, -1)
    scores, inds = torch.topk(heatmaps, K)
    
    clses = (inds // (FM_H * FM_W)).int()
    inds = inds % (FM_H * FM_W)
    ys = (inds // FM_W).int()
    xs = (inds % FM_W).int()
    
    preds = []
    for b in range(batch_size):
        b_preds = []
        for i in range(K):
            score = scores[b, i].item()
            if score < 0.1:
                continue
            cx = xs[b, i].item()
            cy = ys[b, i].item()
            cls_idx = clses[b, i].item()
            
            reg = reg_preds[b, :, cy, cx]
            dx, dy, z, log_w, log_l, log_h, sin_yaw, cos_yaw = reg.tolist()
            
            x = (cx + dx) * STRIDE * BEV_RES + BEV_X_MIN
            y = (cy + dy) * STRIDE * BEV_RES + BEV_Y_MIN
            w, l, h = np.exp(log_w), np.exp(log_l), np.exp(log_h)
            yaw = np.arctan2(sin_yaw, cos_yaw)
            
            cls_name = CLASSES[cls_idx]
            b_preds.append([score, x, y, z, w, l, h, yaw, cls_name])
        preds.append(b_preds)
    return preds

def main():
    print("Loading Dataset...")
    lyft = LyftDataset(data_path='/workspace/lyft_data', json_path='/workspace/lyft_data/train_data', verbose=False)
    
    train_df = pd.read_csv('/workspace/metadata/train_ids.csv')
    val_df = pd.read_csv('/workspace/metadata/val_ids.csv')
    
    train_tokens = train_df['sample_token'].tolist()
    val_tokens = val_df['sample_token'].tolist()
    
    train_dataset = LyftBEVDataset(lyft, train_tokens, is_train=True)
    val_dataset = LyftBEVDataset(lyft, val_tokens, is_train=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = SimpleCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    
    l1_loss = nn.L1Loss(reduction='none')
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for i, (img, hm_gt, reg_gt, reg_mask) in enumerate(train_loader):
            img = img.to(DEVICE)
            hm_gt = hm_gt.to(DEVICE)
            reg_gt = reg_gt.to(DEVICE)
            reg_mask = reg_mask.to(DEVICE)
            
            optimizer.zero_grad()
            hm_pred, reg_pred = model(img)
            
            loss_hm = focal_loss(hm_pred, hm_gt)
            loss_reg = (l1_loss(reg_pred, reg_gt) * reg_mask).sum() / (reg_mask.sum() + 1e-4)
            loss = loss_hm + loss_reg
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch} Step {i}/{len(train_loader)} Loss: {loss.item():.4f} (HM: {loss_hm.item():.4f}, REG: {loss_reg.item():.4f})")
                
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, hm_gt, reg_gt, reg_mask in val_loader:
                img = img.to(DEVICE)
                hm_gt = hm_gt.to(DEVICE)
                reg_gt = reg_gt.to(DEVICE)
                reg_mask = reg_mask.to(DEVICE)
                
                hm_pred, reg_pred = model(img)
                loss_hm = focal_loss(hm_pred, hm_gt)
                loss_reg = (l1_loss(reg_pred, reg_gt) * reg_mask).sum() / (reg_mask.sum() + 1e-4)
                loss = loss_hm + loss_reg
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"Epoch {epoch} Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
    print(f"VAL_METRIC: {-best_val_loss}") # Output negative loss so higher is better
    
    # Inference on Test Set
    print("Inference on Test Set...")
    lyft_test = LyftDataset(data_path='/workspace/lyft_data_test', json_path='/workspace/lyft_data_test/test_data', verbose=False)
    test_df = pd.read_csv('/data/sample_submission.csv')
    test_tokens = test_df['Id'].tolist()
    
    test_dataset = LyftBEVDataset(lyft_test, test_tokens, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    sub_lines = []
    with torch.no_grad():
        for img, tokens in test_loader:
            img = img.to(DEVICE)
            hm_pred, reg_pred = model(img)
            batch_preds = decode_predictions(hm_pred, reg_pred, K=100)
            
            for i, preds in enumerate(batch_preds):
                token = tokens[i]
                
                sample = lyft_test.get('sample', token)
                lidar_token = sample['data']['LIDAR_TOP']
                lidar_data = lyft_test.get('sample_data', lidar_token)
                
                # Transformations Ego -> Global
                ep_record = lyft_test.get('ego_pose', lidar_data['ego_pose_token'])
                ego_rot = Quaternion(ep_record['rotation'])
                ego_trans = np.array(ep_record['translation'])
                
                pred_str = []
                for p in preds:
                    score, x, y, z, w, l, h, yaw, cls_name = p
                    
                    # Point ego -> global
                    pt = np.array([x, y, z])
                    pt = ego_rot.rotation_matrix @ pt + ego_trans
                    
                    # Yaw ego -> global
                    box_rot = Quaternion(axis=[0, 0, 1], angle=yaw)
                    global_rot = ego_rot * box_rot
                    gyaw = global_rot.yaw_pitch_roll[0]
                    
                    pred_str.append(f"{score} {pt[0]} {pt[1]} {pt[2]} {w} {l} {h} {gyaw} {cls_name}")
                
                sub_lines.append(f"{token},{' '.join(pred_str)}")
                
    with open('submission.csv', 'w') as f:
        f.write("Id,PredictionString\n")
        f.write("\n".join(sub_lines) + "\n")
        
if __name__ == '__main__':
    main()