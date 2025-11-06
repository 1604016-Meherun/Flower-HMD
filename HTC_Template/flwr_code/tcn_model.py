import torch
import torch.nn as nn

class TCNModel(nn.Module):
    def __init__(self, C=6, T=64, num_classes=3):
        super().__init__()
        # Replace with your real TCN; keep input [B,C,T]
        self.net = nn.Sequential(
            nn.Conv1d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):              # x: [B,C,T]
        h = self.net(x).squeeze(-1)    # [B,32]
        return self.head(h)            # [B,num_classes]

    def loss_fn(self, logits, y):
        return nn.CrossEntropyLoss()(logits, y)
