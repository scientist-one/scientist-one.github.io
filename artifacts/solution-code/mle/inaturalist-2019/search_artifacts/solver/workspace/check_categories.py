import json
with open("/data/train2019.json", "r") as f:
    data = json.load(f)

print(list(data['categories'][0].keys()))
print(data['categories'][0])
