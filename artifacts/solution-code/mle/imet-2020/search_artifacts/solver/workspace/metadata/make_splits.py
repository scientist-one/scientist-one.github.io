import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def main():
    df = pd.read_csv("/data/train.csv")
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    
    train_ids = train_df[['id']]
    val_ids = val_df[['id']]
    
    train_ids.to_csv("/workspace/metadata/train_ids.csv", index=False)
    val_ids.to_csv("/workspace/metadata/val_ids.csv", index=False)
    print(f"Saved {len(train_ids)} train ids and {len(val_ids)} val ids.")

    with open("/workspace/metadata/README.md", "w") as f:
        f.write("Used random 80:20 split because iterative-stratification on 3474 classes is computationally prohibitive. Random state is fixed to 42.")

if __name__ == "__main__":
    main()
