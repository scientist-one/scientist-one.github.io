import json
import numpy as np
import torch
import torch.nn.functional as F

def compute_taxonomic_similarity():
    with open("/data/train2019.json", "r") as f:
        data = json.load(f)
        
    categories = data['categories']
    categories.sort(key=lambda x: x['id'])
    
    num_classes = len(categories)
    T = np.zeros((num_classes, num_classes))
    
    levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                T[i, j] = 1.0
                continue
            
            shared = 0
            for lvl in levels:
                if categories[i][lvl] == categories[j][lvl]:
                    shared += 1
                else:
                    break
            T[i, j] = shared / len(levels)
            
    return T

if __name__ == "__main__":
    T = compute_taxonomic_similarity()
    print("T shape:", T.shape)
    print("Mean T:", T.mean())
    print("Max non-diag T:", np.max(T - np.eye(len(T))))
    np.save("taxonomic_sim.npy", T)
