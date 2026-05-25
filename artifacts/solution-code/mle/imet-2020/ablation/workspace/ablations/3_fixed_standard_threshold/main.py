import os
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pathlib

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 3474
CACHE_DIR = "/workspace/.cache/my_features"

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

def load_features(split_name, res):
    features_path = os.path.join(CACHE_DIR, f"{split_name}_features_{res}_v2.pt")
    targets_path = os.path.join(CACHE_DIR, f"{split_name}_targets_{res}_v2.pt")
    ids_path = os.path.join(CACHE_DIR, f"{split_name}_ids_{res}_v2.csv")
    
    features = torch.load(features_path)
    targets = torch.load(targets_path) if os.path.exists(targets_path) else None
    ids_df = pd.read_csv(ids_path)
    return features, targets, ids_df

def predict_ensemble(features, model_dir, num_models=10):
    preds_models = np.zeros((num_models, len(features), NUM_CLASSES), dtype=np.float32)
    for m in range(num_models):
        model = MLP().to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(model_dir, f"model_{m}.pt"), map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(features), 1024):
                x = features[i:i+1024].to(DEVICE)
                preds.append(torch.sigmoid(model(x)).cpu().numpy())
            preds_models[m] = np.concatenate(preds)
    return preds_models

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
    # Ensembles
    # 336_v2 -> best ASL model: 81c407b9
    # 448_v2 -> best focal model: ed71bd92
    
    val_features_336, val_targets, _ = load_features('val', 336)
    val_features_448, _, _ = load_features('val', 448)
    
    val_preds_336 = predict_ensemble(val_features_336, '/workspace/solution_81c407b9/')
    val_preds_448 = predict_ensemble(val_features_448, '/workspace/solution_eb839396/')
    
    # We can average all 20 models
    val_preds_ensemble = np.concatenate([val_preds_336, val_preds_448], axis=0).mean(axis=0)
    
    val_targets_np = val_targets.numpy()
    # best_f1, best_thresh = find_best_threshold(val_preds_ensemble, val_targets_np)
    best_thresh = 0.5
    
    # Calculate best_f1 for the hardcoded threshold
    preds_t = (val_preds_ensemble > best_thresh).astype(np.float32)
    tp = (preds_t * val_targets_np).sum()
    fp = (preds_t * (1 - val_targets_np)).sum()
    fn = ((1 - preds_t) * val_targets_np).sum()
    best_f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    print(f"VAL_METRIC: {best_f1}")
    print(f"Best threshold: {best_thresh}")
    
    test_features_336, _, test_ids_df = load_features('test', 336)
    test_features_448, _, _ = load_features('test', 448)
    
    test_preds_336 = predict_ensemble(test_features_336, '/workspace/solution_81c407b9/')
    test_preds_448 = predict_ensemble(test_features_448, '/workspace/solution_ed71bd92/')
    
    test_preds_ensemble = np.concatenate([test_preds_336, test_preds_448], axis=0).mean(axis=0)
    test_preds_bin = (test_preds_ensemble > best_thresh).astype(int)
    
    submission = []
    for i, img_id in enumerate(test_ids_df['id']):
        labels = np.where(test_preds_bin[i] == 1)[0]
        labels_str = " ".join([str(l) for l in labels])
        submission.append({'id': img_id, 'attribute_ids': labels_str})
        
    sub_df = pd.DataFrame(submission)
    sample_sub = pd.read_csv('/data/sample_submission.csv')
    sub_df = sample_sub[['id']].merge(sub_df, on='id', how='left')
    sub_df['attribute_ids'] = sub_df['attribute_ids'].fillna('')
    out_path = pathlib.Path(__file__).resolve().parent / 'submission.csv'
    sub_df.to_csv(out_path, index=False)

if __name__ == '__main__':
    main()
