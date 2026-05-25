import os
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42

def main():
    os.makedirs('metadata', exist_ok=True)
    
    # Read the train split tokens
    train_csv = pd.read_csv('/data/train.csv')
    
    # Let's load sample.json and map sample -> scene
    with open('/data/train_data/sample.json', 'r') as f:
        samples = json.load(f)
    
    sample_to_scene = {s['token']: s['scene_token'] for s in samples}
    
    df = train_csv[['Id']].copy()
    df.rename(columns={'Id': 'sample_token'}, inplace=True)
    df['scene_token'] = df['sample_token'].map(sample_to_scene)
    
    print(f"Total samples: {len(df)}")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    
    for train_idx, val_idx in gss.split(df, groups=df['scene_token']):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
    train_df[['sample_token']].to_csv('metadata/train_ids.csv', index=False)
    val_df[['sample_token']].to_csv('metadata/val_ids.csv', index=False)
    
    with open('metadata/README.md', 'w') as f:
        f.write("Splits created using GroupShuffleSplit on scene_token to prevent data leakage across frames of the same scene. Ratio 80:20, random_state=42.")
        
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

if __name__ == "__main__":
    main()