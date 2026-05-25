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
    
    # Some slices might be corrupt or have different sizes, but usually they are uniform
    # We will filter out ones that don't match the most common shape if needed
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
    # Crop to non-zero
    non_zero = np.nonzero(volume)
    if len(non_zero[0]) == 0:
        return np.zeros(size)
    z_min, z_max = np.min(non_zero[0]), np.max(non_zero[0])
    y_min, y_max = np.min(non_zero[1]), np.max(non_zero[1])
    x_min, x_max = np.min(non_zero[2]), np.max(non_zero[2])
    
    volume = volume[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    
    # Normalize to 0-1
    v_min, v_max = volume.min(), volume.max()
    if v_max > v_min:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = volume - v_min
        
    volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # 1, 1, D, H, W
    volume = F.interpolate(volume, size=size, mode='trilinear', align_corners=False)
    return volume.squeeze(0).squeeze(0).numpy()

class BraTSDataset(Dataset):
    def __init__(self, data_dir, ids, labels=None, modalities=['FLAIR', 'T1w', 'T1wCE', 'T2w'], size=(32, 64, 64), cache_dir=None):
        self.data_dir = data_dir
        self.ids = ids
        self.labels = labels
        self.modalities = modalities
        self.size = size
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        brats_id = self.ids[idx]
        
        # Check cache
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"{brats_id}.npy")
            if os.path.exists(cache_path):
                x = np.load(cache_path)
                x = torch.tensor(x, dtype=torch.float32)
                if self.labels is not None:
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
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y
        return x, brats_id

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels):
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
        
        # size after 4 pools of (32, 64, 64) -> (2, 4, 4)
        self.fc1 = nn.Linear(128 * 2 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x.squeeze(1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    modalities = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    # modalities = ['FLAIR'] # Try with 1 modality to speed up maybe? 
    size = (32, 64, 64)
    
    cache_dir = '/workspace/solution_0e92c8cb/cache'
    train_dataset = BraTSDataset('/data/train', train_ids, train_labels, modalities=modalities, size=size, cache_dir=cache_dir)
    val_dataset = BraTSDataset('/data/train', val_ids, val_labels, modalities=modalities, size=size, cache_dir=cache_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    model = Simple3DCNN(in_channels=len(modalities)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 10
    best_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        auc = roc_auc_score(all_targets, all_preds)
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_model.pth')
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {auc:.4f} | Time: {time.time() - start_time:.1f}s")
        
    print(f"VAL_METRIC: {best_auc}")
    
    # Inference on Test Set
    test_dir = '/data/test'
    test_ids = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
    test_ids.sort()
    
    test_cache_dir = '/workspace/solution_0e92c8cb/cache_test'
    test_dataset = BraTSDataset(test_dir, test_ids, labels=None, modalities=modalities, size=size, cache_dir=test_cache_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    test_preds = []
    test_ids_out = []
    
    with torch.no_grad():
        for x, ids in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.sigmoid(outputs).cpu().numpy()
            test_preds.extend(preds)
            test_ids_out.extend(ids)
            
    submission = pd.DataFrame({
        'BraTS21ID': test_ids_out,
        'MGMT_value': test_preds
    })
    submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
