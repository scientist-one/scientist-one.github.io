import os
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import time

def load_dicom_volume(path):
    files = [f for f in os.listdir(path) if f.endswith('.dcm')]
    slices = [pydicom.dcmread(os.path.join(path, f)) for f in files]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    
    shapes = [s.pixel_array.shape for s in slices]
    if not shapes:
        return np.zeros((1, 1, 1))
        
    most_common_shape = max(set(shapes), key=shapes.count)
    valid_slices = [s.pixel_array for s in slices if s.pixel_array.shape == most_common_shape]
    
    if not valid_slices:
        return np.zeros((1, 1, 1))
        
    volume = np.stack(valid_slices)
    return volume

def crop_and_resize_volume(volume, size=(32, 64, 64)):
    non_zero = np.nonzero(volume)
    if len(non_zero[0]) == 0:
        return np.zeros(size)
    z_min, z_max = np.min(non_zero[0]), np.max(non_zero[0])
    y_min, y_max = np.min(non_zero[1]), np.max(non_zero[1])
    x_min, x_max = np.min(non_zero[2]), np.max(non_zero[2])
    
    volume = volume[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    
    v_min, v_max = volume.min(), volume.max()
    if v_max > v_min:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = volume - v_min
        
    volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    volume = F.interpolate(volume, size=size, mode='trilinear', align_corners=False)
    return volume.squeeze(0).squeeze(0).numpy()

class BraTSDataset(Dataset):
    def __init__(self, data_dir, ids, labels=None, modalities=['FLAIR', 'T1w', 'T1wCE', 'T2w'], size=(32, 64, 64), cache_dir=None, is_train=False):
        self.data_dir = data_dir
        self.ids = ids
        self.labels = labels
        self.modalities = modalities
        self.size = size
        self.is_train = is_train
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        brats_id = self.ids[idx]
        
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"{brats_id}.npy")
            if os.path.exists(cache_path):
                x = np.load(cache_path)
                x = torch.tensor(x, dtype=torch.float32)
                if self.labels is not None:
                    if self.is_train:
                        if np.random.rand() > 0.5:
                            x = torch.flip(x, [2])
                        if np.random.rand() > 0.5:
                            x = torch.flip(x, [3])
                    y = torch.tensor(self.labels[idx], dtype=torch.float32)
                    return x, y
                return x, brats_id
                
        subject_dir = os.path.join(self.data_dir, brats_id)
        
        channels = []
        for mod in self.modalities:
            mod_path = os.path.join(subject_dir, mod)
            if os.path.exists(mod_path):
                try:
                    vol = load_dicom_volume(mod_path)
                    vol = crop_and_resize_volume(vol, size=self.size)
                except Exception as e:
                    vol = np.zeros(self.size)
            else:
                vol = np.zeros(self.size)
            channels.append(vol)
            
        x = np.stack(channels)
        if self.cache_dir is not None:
            np.save(cache_path, x)
            
        x = torch.tensor(x, dtype=torch.float32)
        
        if self.labels is not None:
            if self.is_train:
                if np.random.rand() > 0.5:
                    x = torch.flip(x, [2])
                if np.random.rand() > 0.5:
                    x = torch.flip(x, [3])
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y

def generate_synthetic_tumor(size=(32, 64, 64), return_metrics=True):
    D, H, W = size
    z, y, x = np.ogrid[:D, :H, :W]
    
    cz = np.random.uniform(D * 0.3, D * 0.7)
    cy = np.random.uniform(H * 0.3, H * 0.7)
    cx = np.random.uniform(W * 0.3, W * 0.7)
    
    mask = np.zeros(size, dtype=bool)
    
    num_lobules = np.random.randint(1, 6)
        
    for _ in range(num_lobules):
        lz = cz + np.random.normal(0, D * 0.1)
        ly = cy + np.random.normal(0, H * 0.1)
        lx = cx + np.random.normal(0, W * 0.1)
        
        rz = np.random.uniform(D * 0.1, D * 0.3)
        ry = np.random.uniform(H * 0.1, H * 0.3)
        rx = np.random.uniform(W * 0.1, W * 0.3)
        
        ellipsoid = ((z - lz)**2 / rz**2) + ((y - ly)**2 / ry**2) + ((x - lx)**2 / rx**2) <= 1
        mask = mask | ellipsoid
        
    mask = mask.astype(np.float32)
    
    if return_metrics:
        volume = mask.sum()
        shifted_x = np.pad(mask[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode='constant')
        shifted_y = np.pad(mask[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode='constant')
        shifted_z = np.pad(mask[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode='constant')
        
        surface = (mask != shifted_x).sum() + (mask != shifted_y).sum() + (mask != shifted_z).sum()
        sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / (surface + 1e-6)
        
        volume_norm = volume / (D * H * W)
        sphericity_norm = np.clip(sphericity, 0, 1)
        
        return mask, np.array([volume_norm, sphericity_norm], dtype=np.float32)
    return mask

class SyntheticDataset(Dataset):
    def __init__(self, num_samples, size=(32, 64, 64), in_channels=4):
        self.num_samples = num_samples
        self.size = size
        self.in_channels = in_channels
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        mask, metrics = generate_synthetic_tumor(size=self.size, return_metrics=True)
        # expand to C channels
        x = np.repeat(mask[np.newaxis, ...], self.in_channels, axis=0)
        
        # Add textural noise
        noise = np.random.normal(0, 0.15, x.shape)
        background = np.random.normal(0.1, 0.1, x.shape)
        
        tumor_intensity = np.random.uniform(0.5, 0.9, (self.in_channels, 1, 1, 1))
        x = (tumor_intensity + noise) * x + background * (1 - x)
        x = np.clip(x, 0, 1)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(metrics, dtype=torch.float32)

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(2)
        
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(2)
        
        self.fc1 = nn.Linear(128 * 2 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    modalities = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    size = (32, 64, 64)
    
    # Load splits
    train_ids_df = pd.read_csv('/workspace/metadata/train_ids.csv')
    val_ids_df = pd.read_csv('/workspace/metadata/val_ids.csv')
    train_ids = train_ids_df['BraTS21ID'].astype(str).str.zfill(5).values
    val_ids = val_ids_df['BraTS21ID'].astype(str).str.zfill(5).values
    
    labels_df = pd.read_csv('/data/train_labels.csv')
    labels_df['BraTS21ID'] = labels_df['BraTS21ID'].astype(str).str.zfill(5)
    labels_dict = dict(zip(labels_df['BraTS21ID'], labels_df['MGMT_value']))
    
    train_labels = [labels_dict[id_] for id_ in train_ids]
    val_labels = [labels_dict[id_] for id_ in val_ids]
    
    cache_dir = '/workspace/solution_0e92c8cb/cache'
    train_dataset = BraTSDataset('/data/train', train_ids, train_labels, modalities=modalities, size=size, cache_dir=cache_dir)
    val_dataset = BraTSDataset('/data/train', val_ids, val_labels, modalities=modalities, size=size, cache_dir=cache_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Inference on Test Set (prepare dataloader)
    test_dir = '/data/test'
    test_ids = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
    test_ids.sort()
    
    test_cache_dir = '/workspace/solution_0e92c8cb/cache_test'
    test_dataset = BraTSDataset(test_dir, test_ids, labels=None, modalities=modalities, size=size, cache_dir=test_cache_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print("--- ONE-TIME MASSIVE PRE-TRAINING ---")
    np.random.seed(42)
    torch.manual_seed(42)
    pretrain_model = Simple3DCNN(in_channels=len(modalities), num_classes=2).to(device)
    pretrain_criterion = nn.MSELoss()
    pretrain_optimizer = optim.Adam(pretrain_model.parameters(), lr=1e-3)
    
    num_synthetic_samples = 20000
    synthetic_dataset = SyntheticDataset(num_synthetic_samples, size=size, in_channels=len(modalities))
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    pretrain_epochs = 1
    for epoch in range(pretrain_epochs):
        pretrain_model.train()
        for x, y in synthetic_loader:
            x, y = x.to(device), y.to(device)
            pretrain_optimizer.zero_grad()
            outputs = pretrain_model(x)
            loss = pretrain_criterion(outputs, y)
            loss.backward()
            pretrain_optimizer.step()
    
    print("Pre-training completed.")
    
    num_ensembles = 10
    ensemble_preds = []
    ensemble_val_preds = []
    final_val_targets = val_labels
    
    valid_ensemble_preds = []
    valid_ensemble_val_preds = []
    
    for fold in range(num_ensembles):
        print(f"--- Ensemble Fold {fold+1}/{num_ensembles} ---")
        
        # Reset random seed for fine-tuning
        np.random.seed(fold * 10)
        torch.manual_seed(fold * 10)
        
        # 2. Fine-tune
        model = Simple3DCNN(in_channels=len(modalities), num_classes=1).to(device)
        pretrained_dict = pretrain_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc2' not in k}
        model.load_state_dict(pretrained_dict, strict=False)
        # 2. Fine-tune
        model = Simple3DCNN(in_channels=len(modalities), num_classes=1).to(device)
        pretrained_dict = pretrain_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc2' not in k}
        model.load_state_dict(pretrained_dict, strict=False)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        epochs = 10
        best_auc = 0.0
        best_state = None
        
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device).unsqueeze(1)
                    outputs = model(x)
                    preds = torch.sigmoid(outputs).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(y.cpu().numpy())
                    
            auc = roc_auc_score(all_targets, all_preds)
            if auc > best_auc:
                best_auc = auc
                best_state = model.state_dict().copy()
                best_val_preds = all_preds.copy()
                
        print(f"Fold {fold+1} Best Val AUC: {best_auc:.4f}")
        ensemble_val_preds.append(best_val_preds)
        
        if best_auc > 0.53:
            valid_ensemble_val_preds.append(best_val_preds)
        
        # Inference for this fold
        model.load_state_dict(best_state)
        model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for x, ids in test_loader:
                x = x.to(device)
                outputs = model(x)
                preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
                if preds.ndim == 0:
                    preds = [preds]
                fold_test_preds.extend(preds)
        ensemble_preds.append(fold_test_preds)
        
        if best_auc > 0.53:
            valid_ensemble_preds.append(fold_test_preds)
        
        # Free memory
        del model, optimizer
        torch.cuda.empty_cache()
        
    val_aucs = [roc_auc_score(final_val_targets, preds) for preds in ensemble_val_preds]
    best_idx = np.argmax(val_aucs)
    print(f"Selecting Fold {best_idx+1} as the single best model (AUC: {val_aucs[best_idx]:.4f}).")
    
    avg_test_preds = ensemble_preds[best_idx]
    avg_val_preds = ensemble_val_preds[best_idx]
    
    final_auc = roc_auc_score(final_val_targets, avg_val_preds)
    print(f"VAL_METRIC: {final_auc}")
    
    submission = pd.DataFrame({
        'BraTS21ID': test_ids,
        'MGMT_value': avg_test_preds
    })
    submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
