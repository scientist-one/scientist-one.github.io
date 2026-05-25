import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def main():
    os.makedirs("metadata", exist_ok=True)
    
    with open("/data/train2019.json", "r") as f:
        data = json.load(f)
        
    # Create dataframe of annotations
    annotations = data['annotations']
    df = pd.DataFrame(annotations)
    
    # We want to stratify by category_id
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=RANDOM_STATE, 
        stratify=df['category_id']
    )
    
    train_ids = train_df[['image_id']].copy()
    val_ids = val_df[['image_id']].copy()
    
    train_ids.to_csv("metadata/train_ids.csv", index=False)
    val_ids.to_csv("metadata/val_ids.csv", index=False)
    
    with open("metadata/README.md", "w") as f:
        f.write("Stratified 80:20 split by `category_id` using train2019.json, since val2019.json has missing images.\n")
        f.write("Seed: 42.\n")
        f.write("ID schema: `image_id` matches the `id` field in images list of train2019.json.\n")
    
    print(f"Train size: {len(train_ids)}, Val size: {len(val_ids)}")

if __name__ == "__main__":
    main()
