# flwr_time_series_client.py
from datetime import datetime

from flwr.client import NumPyClient, ClientApp

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import train_test_split

# ----------------------------
# Config: point this to your Unity CSV
# ----------------------------
CSV_PATH = r"C:\Users\meher\AppData\LocalLow\DefaultCompany\HTC_Template\client1\Fixation_20250929_120113_01aa2306-a174-49b0-86ab-618d7b33e692.csv"
TARGET_COL = "SceneName"          # label column
TEST_SIZE  = 0.2
BATCH_SIZE = 32
WINDOW     = 120                  # e.g., 1 second if 120 Hz
STRIDE     = 60                   # 50% overlap
LR         = 1e-3
EPOCHS_PER_FIT = 1                # per FL round

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Data loading & windowing
# ----------------------------
def df_to_windows(df: pd.DataFrame, target_col: str, window: int, stride: int):
    """
    Converts a time-ordered dataframe into sliding windows.
    Features: all numeric columns except target_col.
    Label per window: majority vote of target_col in that window (or final row).
    """
    assert target_col in df.columns, f"'{target_col}' not found in CSV"
    # Keep only numeric feature columns + target
    feature_cols = [c for c in df.columns if c != target_col]
    # Use only numeric features; drop non-numeric gracefully
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("No numeric feature columns found besides the target.")
    X_all = df[num_cols].to_numpy(dtype=np.float32)

    # Encode labels (factorize preserves order of appearance)
    labels, uniques = pd.factorize(df[target_col], sort=True)
    y_all = labels.astype(np.int64)
    label_map = {label: idx for idx, label in enumerate(uniques.tolist())}
    inv_label_map = {v: k for k, v in label_map.items()}

    n = len(df)
    X_windows = []
    y_windows = []

    # Slide windows
    i = 0
    while i + window <= n:
        xw = X_all[i:i+window]               # shape [window, feat_dim]
        yw_slice = y_all[i:i+window]         # labels over the window
        # Robust window label: majority vote (fallback to last label)
        if len(yw_slice) > 0:
            vals, counts = np.unique(yw_slice, return_counts=True)
            y_win = int(vals[np.argmax(counts)])
        else:
            y_win = int(y_all[min(i+window-1, n-1)])
        X_windows.append(xw)
        y_windows.append(y_win)
        i += stride

    X = np.stack(X_windows, axis=0)          # [num_windows, window, feat_dim]
    y = np.array(y_windows, dtype=np.int64)  # [num_windows]

    # Debug: duplicate data to increase size
    X = np.concatenate((X, X), axis=0)
    y = np.concatenate((y, y), axis=0)

    #X and Y shapes and dtypes
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")

    # Memory size (bytes → MB)
    X_mem_mb = X.nbytes / (1024**2)
    y_mem_mb = y.nbytes / (1024**2)

    print(f"X memory: {X_mem_mb:.2f} MB")
    print(f"y memory: {y_mem_mb:.2f} MB")

    return X, y, num_cols, inv_label_map


def load_ts_data(csv_path: str, target_col: str, window: int, stride: int, test_size: float):
    # If there's a timestamp column, sort by it; otherwise keep original order
    df = pd.read_csv(csv_path)
    ts_candidates = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    if ts_candidates:
        df = df.sort_values(by=ts_candidates[0])

    X, y, feat_cols, inv_label_map = df_to_windows(df, target_col, window, stride)

    # Train-test split by windows
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y
    )

    # PyTorch tensors
    # For Conv1d: input is [batch, channels, seq_len] -> channels = feat_dim, seq_len = window
    # Current X is [N, window, feat_dim], so we transpose last two dims
    X_train_t = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)
    X_test_t  = torch.tensor(X_test , dtype=torch.float32).transpose(1, 2)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t  = torch.tensor(y_test , dtype=torch.long)

    num_classes = int(y.max()) + 1
    feat_dim    = X_train_t.shape[1]
    seq_len     = X_train_t.shape[2]

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds  = TensorDataset(X_test_t , y_test_t)

    return train_ds, test_ds, num_classes, feat_dim, seq_len, inv_label_map


# ----------------------------
# Time-series model (1D-CNN)
# ----------------------------
class CNN1DClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # Input shape: [B, C=in_channels, T]
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1)  # -> [B, 128, 1]
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x)           # [B, 128, 1]
        x = x.squeeze(-1)         # [B, 128]
        logits = self.head(x)     # [B, num_classes]
        return logits


# ----------------------------
# Train / Eval
# ----------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total += xb.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total += xb.size(0)

    return total_loss / total, total_correct / total


# ----------------------------
# Prepare data / model
# ----------------------------
train_ds, test_ds, NUM_CLASSES, FEAT_DIM, SEQ_LEN, INV_LABEL_MAP = load_ts_data(
    CSV_PATH, TARGET_COL, WINDOW, STRIDE, TEST_SIZE
)

trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
testloader  = DataLoader(test_ds , batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

net = CNN1DClassifier(in_channels=FEAT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

print(f"[Info] Features: {FEAT_DIM}, SeqLen: {SEQ_LEN}, Classes: {NUM_CLASSES}")
print(f"[Info] Label map: {INV_LABEL_MAP}")


# ----------------------------
# Flower client
# ----------------------------
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Train for EPOCHS_PER_FIT each FL round
        for _ in range(EPOCHS_PER_FIT):
            train_loss, train_acc = train_epoch(net, trainloader, optimizer, criterion)
        return self.get_parameters(config={}), len(trainloader.dataset), {"train_loss": train_loss, "train_acc": train_acc}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, val_acc = evaluate(net, testloader, criterion)
        return float(val_loss), len(testloader.dataset), {"accuracy": float(val_acc)}


def client_fn(cid: str):
    return FlowerClient().to_client()


app = ClientApp(client_fn=client_fn)


# ----------------------------
# Legacy start (direct client)
# ----------------------------
# if __name__ == "__main__":
#     from flwr.client import start_client
#     start_client(
#         server_address="127.0.0.1:5006",
#         client=FlowerClient().to_client(),
#     )


if __name__ == "__main__":
    from flwr.client import start_client
    
    start_utc = datetime.utcnow()
    print(f"[UTC START] {start_utc.isoformat()}Z")

    start_client(
        server_address="127.0.0.1:5006",
        client=FlowerClient().to_client(),
    )

    end_utc = datetime.utcnow()
    print(f"[UTC END]   {end_utc.isoformat()}Z")
    print(f"[DURATION]  {end_utc - start_utc}")
