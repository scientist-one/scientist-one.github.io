import json
import pandas as pd
import os

df_orders = pd.read_csv('/data/train_orders.csv')
print(f"Total train orders: {len(df_orders)}")

df_ancestors = pd.read_csv('/data/train_ancestors.csv')
print(f"Total ancestors: {len(df_ancestors)}")

train_id = df_orders['id'].iloc[0]
with open(f"/data/train/{train_id}.json", "r") as f:
    notebook = json.load(f)

print(f"Keys in notebook JSON: {notebook.keys()}")
print(f"Cell types: {notebook['cell_type']}")
print(f"Number of cells: {len(notebook['cell_type'])}")
print(f"Number of code cells: {list(notebook['cell_type'].values()).count('code')}")
print(f"Number of markdown cells: {list(notebook['cell_type'].values()).count('markdown')}")

print("\nFirst cell ID and content:")
first_cell_id = list(notebook['cell_type'].keys())[0]
print(f"ID: {first_cell_id}")
print(f"Type: {notebook['cell_type'][first_cell_id]}")
print(f"Source: {notebook['source'][first_cell_id][:100]}")
