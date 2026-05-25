import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from tqdm import tqdm
from bisect import bisect
import time

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

def read_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_ranks(cell_order_str, cell_types):
    cell_order = cell_order_str.split()
    
    code_count = sum(1 for c in cell_order if cell_types[c] == 'code')
    
    y = {}
    current_code_idx = 0
    
    for cell_id in cell_order:
        if cell_types[cell_id] == 'code':
            current_code_idx += 1
        else:
            if code_count > 0:
                y[cell_id] = current_code_idx / code_count
            else:
                y[cell_id] = 0.5
                
    return y

def load_data(ids, orders_df=None):
    texts = []
    targets = []
    notebook_cell_ids = []
    
    orders_dict = {}
    if orders_df is not None:
        orders_dict = dict(zip(orders_df['id'], orders_df['cell_order']))
        
    for nb_id in tqdm(ids, desc="Loading Data"):
        nb = read_notebook(f'/data/train/{nb_id}.json')
        cell_types = nb['cell_type']
        source = nb['source']
        
        y = {}
        if nb_id in orders_dict:
            y = get_ranks(orders_dict[nb_id], cell_types)
            
        nb_cells = []
        for cell_id, c_type in cell_types.items():
            if c_type == 'markdown':
                texts.append(source[cell_id])
                targets.append(y.get(cell_id, 0.0))
                nb_cells.append(cell_id)
                
        notebook_cell_ids.append((nb_id, nb_cells))
        
    return texts, targets, notebook_cell_ids

def main():
    print("Loading splits...")
    train_ids = pd.read_csv('/workspace/metadata/train_ids.csv')['id'].values
    val_ids = pd.read_csv('/workspace/metadata/val_ids.csv')['id'].values
    orders_df = pd.read_csv('/data/train_orders.csv')
    
    # Subsample for faster baseline
    train_ids = train_ids[:5000]
    # Keep full val
    
    print("Loading train data...")
    train_texts, train_targets, _ = load_data(train_ids, orders_df)
    
    print("Loading val data...")
    val_texts, val_targets, val_nb_cells = load_data(val_ids, orders_df)
    
    print("Training TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    
    print("Training Ridge...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, train_targets)
    
    print("Predicting Val...")
    X_val = vectorizer.transform(val_texts)
    val_preds = model.predict(X_val)
    
    # Construct val predictions
    print("Evaluating...")
    pred_idx = 0
    val_predictions_list = []
    val_gt_list = []
    
    orders_dict = dict(zip(orders_df['id'], orders_df['cell_order']))
    
    for nb_id, md_cells in tqdm(val_nb_cells, desc="Reconstructing Val"):
        nb = read_notebook(f'/data/train/{nb_id}.json')
        cell_types = nb['cell_type']
        
        # Get code cells in original order
        code_cells = [c for c, t in cell_types.items() if t == 'code']
        code_count = len(code_cells)
        
        cell_ranks = {}
        # Assign ranks to code cells
        for i, c in enumerate(code_cells):
            cell_ranks[c] = (i + 0.5) / max(code_count, 1)
            
        # Assign predicted ranks to md cells
        for c in md_cells:
            cell_ranks[c] = val_preds[pred_idx]
            pred_idx += 1
            
        # Sort cells
        pred_order = sorted(cell_ranks.keys(), key=lambda x: cell_ranks[x])
        val_predictions_list.append(pred_order)
        
        gt_order = orders_dict[nb_id].split()
        val_gt_list.append(gt_order)
        
    val_score = kendall_tau(val_gt_list, val_predictions_list)
    print(f"VAL_METRIC: {val_score}")
    
    # Predict Test
    print("Predicting Test...")
    test_ids = [f.split('.')[0] for f in os.listdir('/data/test') if f.endswith('.json')]
    test_texts = []
    test_nb_cells = []
    
    for nb_id in test_ids:
        nb = read_notebook(f'/data/test/{nb_id}.json')
        md_cells = [c for c, t in nb['cell_type'].items() if t == 'markdown']
        for c in md_cells:
            test_texts.append(nb['source'][c])
        test_nb_cells.append((nb_id, md_cells))
        
    if test_texts:
        X_test = vectorizer.transform(test_texts)
        test_preds = model.predict(X_test)
    else:
        test_preds = []
        
    sub_rows = []
    pred_idx = 0
    for nb_id, md_cells in test_nb_cells:
        nb = read_notebook(f'/data/test/{nb_id}.json')
        code_cells = [c for c, t in nb['cell_type'].items() if t == 'code']
        code_count = len(code_cells)
        
        cell_ranks = {}
        for i, c in enumerate(code_cells):
            cell_ranks[c] = (i + 0.5) / max(code_count, 1)
            
        for c in md_cells:
            cell_ranks[c] = test_preds[pred_idx]
            pred_idx += 1
            
        pred_order = sorted(cell_ranks.keys(), key=lambda x: cell_ranks[x])
        sub_rows.append({'id': nb_id, 'cell_order': ' '.join(pred_order)})
        
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv('submission.csv', index=False)
    print("Saved submission.csv")

if __name__ == '__main__':
    main()
