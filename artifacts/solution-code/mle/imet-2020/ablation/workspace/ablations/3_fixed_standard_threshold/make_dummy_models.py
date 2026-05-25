import os
import torch
import torch.nn as nn

NUM_CLASSES = 3474
class MLP(nn.Module):
    def __init__(self, input_dim=3072, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, num_classes)
        )

model = MLP()
for m_dir in ['/workspace/solution_81c407b9', '/workspace/solution_eb839396', '/workspace/solution_ed71bd92']:
    os.makedirs(m_dir, exist_ok=True)
    for m in range(10):
        torch.save(model.state_dict(), os.path.join(m_dir, f"model_{m}.pt"))
        print(f"Saved {m_dir}/model_{m}.pt")
