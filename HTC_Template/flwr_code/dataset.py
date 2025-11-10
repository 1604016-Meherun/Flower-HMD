# dataset.py
import torch
from torch.utils.data import Dataset

class QueueSnapshotDataset(Dataset):
    def __init__(self, items, dtype=torch.float32):
        if len(items) == 0:
            self.X = torch.zeros((1, 31), dtype=dtype)  # fallback shape
            self.y = torch.zeros((1,), dtype=torch.long)
        else:
            self.X = torch.tensor([x for x,_ in items], dtype=dtype)
            self.y = torch.tensor([y for _,y in items], dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]
