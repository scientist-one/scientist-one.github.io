import json
import os

with open("/data/val2019.json", "r") as f:
    val_data = json.load(f)

print(f"Number of val images in JSON: {len(val_data['images'])}")
missing = 0
for img in val_data['images']:
    path = os.path.join("/data", img['file_name'])
    if not os.path.exists(path):
        missing += 1

print(f"Missing val images: {missing}")

with open("/data/train2019.json", "r") as f:
    train_data = json.load(f)
print(f"Number of train images in JSON: {len(train_data['images'])}")

missing_train = 0
for img in train_data['images']:
    path = os.path.join("/data", img['file_name'])
    if not os.path.exists(path):
        missing_train += 1
print(f"Missing train images: {missing_train}")
