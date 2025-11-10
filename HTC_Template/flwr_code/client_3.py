# client_3_timeseries.py
from flwr.client import NumPyClient, ClientApp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import train_test_split

# =========================
# Config
# =========================
# CSV_PATH = r"/mnt/data/Fixation_20250728_222559_78a8ef1e-1da3-4984-83b5-68ab36a975f5.csv"
# Example Windows path if running locally:
CSV_PATH = r"C:\Users\meher\AppData\LocalLow\DefaultCompany\HTC_Template\client1\Fixation_20250728_222559_78a8ef1e-1da3-4984-83b5-68ab36a975f5.csv"

TARGET_COL = "SceneName"
TEST_SIZE  = 0.2
BATCH_SIZE = 32

# With 120 Hz eye-tracking, 120 ≈ 1s window
WINDOW     = 120
STRIDE     = 60

LR         = 1e-3
EPOCHS_PER_FIT = 1
SERVER_ADDRESS = "127.0.0.1:5006"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Data pipeline (CSV -> windows)
# =========================
def df_to_windows(df: pd.DataFrame, target_col: str, window: int, stride: int):
    assert target_col in df.columns, f"'{target_col}' not found; columns: {list(df.columns)}"

    # numeric features only (excluding target)
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("No numeric feature columns found; check the CSV.")

    X_all = df[num_cols].to_numpy(dtype=np.float32)

    # encode labels from SceneName
    labels, uniques = pd.factorize(df[target_col], sort=True)
    y_all = labels.astype(np.int64)
    inv_label_map = {i: name for i, name in enumerate(uniques.tolist())}

    # sliding windows
    Xw, yw = [], []
    n = len(df)
    i = 0
    while i + window <= n:
        x_slice = X_all[i:i+window]            # [T, C]
        y_slice = y_all[i:i+window]
        # label by majority vote within the window
        vals, counts = np.unique(y_slice, return_counts=True)
        y_win = int(vals[np.argmax(counts)])
        Xw.append(x_slice)
        yw.append(y_win)
        i += stride

    X = np.stack(Xw, axis=0)                   # [N, T, C]
    y = np.asarray(yw, dtype=np.int64)         # [N]
    return X, y, num_cols, inv_label_map


def load_ts_data(csv_path: str, target_col: str, window: int, stride: int, test_size: float):
    df = pd.read_csv(csv_path)

    # sort by timestamp if present
    ts_cols = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    if ts_cols:
        df = df.sort_values(ts_cols[0])

    X, y, feat_cols, inv_label_map = df_to_windows(df, target_col, window, stride)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y
    )

    # Conv1d expects [B, C, T]; we currently have [N, T, C]
    X_train = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)
    X_test  = torch.tensor(X_test , dtype=torch.float32).transpose(1, 2)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test , dtype=torch.long)

    num_classes = int(y.max()) + 1
    feat_dim    = X_train.shape[1]
    seq_len     = X_train.shape[2]

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test , y_test)

    return train_ds, test_ds, num_classes, feat_dim, seq_len, inv_label_map


# =========================
# 1D-CNN time-series classifier
# =========================
class CNN1DClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
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
        x = self.net(x)       # [B, 128, 1]
        x = x.squeeze(-1)     # [B, 128]
        return self.head(x)   # logits


# =========================
# Train / Eval
# =========================
def train_epoch(model, loader, optimizer, criterion, device=DEVICE):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device=DEVICE):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, total_correct / total


# =========================
# Prep data / loaders / model
# =========================
train_ds, test_ds, NUM_CLASSES, FEAT_DIM, SEQ_LEN, INV_LABEL_MAP = load_ts_data(
    CSV_PATH, TARGET_COL, WINDOW, STRIDE, TEST_SIZE
)

trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
testloader  = DataLoader(test_ds , batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

net = CNN1DClassifier(in_channels=FEAT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

print(f"[client_3] CSV: {CSV_PATH}")
print(f"[client_3] Features: {FEAT_DIM}, SeqLen: {SEQ_LEN}, Classes: {NUM_CLASSES}")
print(f"[client_3] Label map (id -> name): {INV_LABEL_MAP}")


# =========================
# Flower client
# =========================
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [v.detach().cpu().numpy() for _, v in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loss, train_acc = train_epoch(net, trainloader, optimizer, criterion)
        return self.get_parameters(config={}), len(trainloader.dataset), {
            "train_loss": float(train_loss), "train_acc": float(train_acc)
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, val_acc = evaluate(net, testloader, criterion)
        return float(val_loss), len(testloader.dataset), {"accuracy": float(val_acc)}


def client_fn(cid: str):
    return FlowerClient().to_client()


app = ClientApp(client_fn=client_fn)


# =========================
# Legacy start
# =========================
if __name__ == "__main__":
    from flwr.client import start_client
    start_client(
        server_address=SERVER_ADDRESS,
        client=FlowerClient().to_client(),
    )
