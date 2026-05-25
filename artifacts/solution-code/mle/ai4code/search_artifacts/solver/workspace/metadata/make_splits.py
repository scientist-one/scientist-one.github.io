import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os

RANDOM_STATE = 42

def make_splits():
    os.makedirs('/workspace/metadata', exist_ok=True)
    
    # Load orders and ancestors
    df_orders = pd.read_csv('/data/train_orders.csv')
    df_ancestors = pd.read_csv('/data/train_ancestors.csv')
    
    # Merge to get ancestor_id for grouping
    df = df_orders.merge(df_ancestors[['id', 'ancestor_id']], on='id', how='left')
    
    # GroupShuffleSplit based on ancestor_id
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss.split(df, groups=df['ancestor_id']))
    
    train_ids = df.iloc[train_idx]['id'].reset_index(drop=True)
    val_ids = df.iloc[val_idx]['id'].reset_index(drop=True)
    
    train_ids.to_csv('/workspace/metadata/train_ids.csv', index=False, header=['id'])
    val_ids.to_csv('/workspace/metadata/val_ids.csv', index=False, header=['id'])
    
    # Create README.md
    with open('/workspace/metadata/README.md', 'w') as f:
        f.write("# Train/Val Split\n\n")
        f.write("We use an 80:20 train/validation split.\n")
        f.write("The split is performed using GroupShuffleSplit on `ancestor_id` ")
        f.write("to prevent data leakage from forked notebooks.\n")
        f.write(f"Random state is fixed to {RANDOM_STATE}.\n")
        
    print(f"Split created. Train size: {len(train_ids)}, Val size: {len(val_ids)}")

if __name__ == "__main__":
    make_splits()
