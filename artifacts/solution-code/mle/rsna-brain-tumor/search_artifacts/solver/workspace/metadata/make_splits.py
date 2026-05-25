import pandas as pd
import os
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def main():
    metadata_dir = 'metadata'
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Read labels
    df = pd.read_csv('/data/train_labels.csv')
    df['BraTS21ID'] = df['BraTS21ID'].astype(str).str.zfill(5)
    
    # Remove bad cases according to competition description
    bad_cases = ['00109', '00123', '00709']
    df = df[~df['BraTS21ID'].isin(bad_cases)].reset_index(drop=True)
    
    # Stratified split based on MGMT_value
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df['MGMT_value']
    )
    
    # Write to metadata dir
    train_df[['BraTS21ID']].to_csv(os.path.join(metadata_dir, 'train_ids.csv'), index=False)
    val_df[['BraTS21ID']].to_csv(os.path.join(metadata_dir, 'val_ids.csv'), index=False)
    
    print(f"Total valid cases: {len(df)}")
    print(f"Train cases: {len(train_df)}")
    print(f"Val cases: {len(val_df)}")
    print("Class distribution in train:", train_df['MGMT_value'].value_counts().to_dict())
    print("Class distribution in val:", val_df['MGMT_value'].value_counts().to_dict())
    
    # Write README
    readme_content = """# Train/Val Split

This directory contains the fixed train and validation splits for the competition.
- `train_ids.csv`: IDs of the training set.
- `val_ids.csv`: IDs of the validation set.

Rules followed:
- Random state: 42
- Ratio: 80% train, 20% validation
- Strategy: Stratified sampling based on target label `MGMT_value` to ensure class balance across splits.
- Filtered cases: Excluded 00109, 00123, 00709 due to known issues (per competition data description).
- ID column: `BraTS21ID`, formatted as a 5-digit string with leading zeros.
"""
    with open(os.path.join(metadata_dir, 'README.md'), 'w') as f:
        f.write(readme_content)

if __name__ == '__main__':
    main()
