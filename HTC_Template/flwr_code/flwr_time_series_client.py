# # flwr_time_series_client.py — unified version for Unity → Flower

# from datetime import datetime
# import argparse
# import os
# from collections import OrderedDict

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from flwr.client import NumPyClient, start_client


# # ------------------------------------------------------------
# # 1️⃣  Command-line arguments
# # ------------------------------------------------------------
# def parse_args():
#     p = argparse.ArgumentParser(description="Flower client for Unity-generated CSV data")
#     p.add_argument("--csv", required=True, help="Path to the Unity-exported CSV")
#     p.add_argument("--server", default="127.0.0.1:5006", help="Flower server host:port")
#     p.add_argument("--target", default="SceneName", help="Target label column name")
#     p.add_argument("--batch", type=int, default=32)
#     p.add_argument("--window", type=int, default=120)
#     p.add_argument("--stride", type=int, default=60)
#     p.add_argument("--epochs_per_fit", type=int, default=1)
#     p.add_argument("--lr", type=float, default=1e-3)
#     p.add_argument("--test_size", type=float, default=0.2)
#     return p.parse_args()


# ARGS = parse_args()
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ------------------------------------------------------------
# # 2️⃣  Data preparation
# # ------------------------------------------------------------
# def df_to_windows(df: pd.DataFrame, target_col: str, window: int, stride: int):
#     assert target_col in df.columns, f"'{target_col}' not found in CSV"

#     feature_cols = [c for c in df.columns if c != target_col]
#     num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
#     if len(num_cols) == 0:
#         raise ValueError("No numeric feature columns found besides the target.")
#     X_all = df[num_cols].to_numpy(dtype=np.float32)

#     labels, uniques = pd.factorize(df[target_col], sort=True)
#     y_all = labels.astype(np.int64)
#     inv_label_map = {i: cls for i, cls in enumerate(uniques.tolist())}

#     n = len(df)
#     X_windows, y_windows = [], []
#     i = 0
#     while i + window <= n:
#         xw = X_all[i:i + window]
#         yw_slice = y_all[i:i + window]
#         vals, counts = np.unique(yw_slice, return_counts=True)
#         y_win = int(vals[np.argmax(counts)]) if len(vals) else int(y_all[min(i + window - 1, n - 1)])
#         X_windows.append(xw)
#         y_windows.append(y_win)
#         i += stride

#     X = np.stack(X_windows, axis=0)
#     y = np.array(y_windows, dtype=np.int64)

#     print(f"[Data] X shape: {X.shape}, y shape: {y.shape}")
#     print(f"[Data] Features per sample: {X.shape[2]}, window: {window}, stride: {stride}")
#     print(f"[Data] Memory: {X.nbytes/1024**2:.2f} MB")

#     return X, y, num_cols, inv_label_map


# def load_ts_data(csv_path: str, target_col: str, window: int, stride: int, test_size: float):
#     df = pd.read_csv(csv_path)
#     ts_candidates = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
#     if ts_candidates:
#         df = df.sort_values(by=ts_candidates[0])

#     X, y, feat_cols, inv_label_map = df_to_windows(df, target_col, window, stride)
#     stratify_arg = y if len(np.unique(y)) > 1 else None
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=42, shuffle=True, stratify=stratify_arg
#     )

#     # transpose to [B, C, T] for Conv1D
#     X_train_t = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)
#     X_test_t = torch.tensor(X_test, dtype=torch.float32).transpose(1, 2)
#     y_train_t = torch.tensor(y_train, dtype=torch.long)
#     y_test_t = torch.tensor(y_test, dtype=torch.long)

#     num_classes = int(y.max()) + 1
#     feat_dim = X_train_t.shape[1]
#     seq_len = X_train_t.shape[2]

#     train_ds = TensorDataset(X_train_t, y_train_t)
#     test_ds = TensorDataset(X_test_t, y_test_t)
#     return train_ds, test_ds, num_classes, feat_dim, seq_len, inv_label_map


# # ------------------------------------------------------------
# # 3️⃣  Model definition
# # ------------------------------------------------------------
# class CNN1DClassifier(nn.Module):
#     def __init__(self, in_channels: int, num_classes: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(64, 128, kernel_size=5, padding=2),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool1d(1),
#         )
#         self.head = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.net(x)
#         x = x.squeeze(-1)
#         return self.head(x)


# # ------------------------------------------------------------
# # 4️⃣  Train / eval helpers
# # ------------------------------------------------------------
# def train_epoch(model, loader, optimizer, criterion):
#     model.train()
#     total_loss, total_correct, total = 0.0, 0, 0
#     for xb, yb in loader:
#         xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#         optimizer.zero_grad()
#         logits = model(xb)
#         loss = criterion(logits, yb)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * xb.size(0)
#         total_correct += (logits.argmax(1) == yb).sum().item()
#         total += xb.size(0)
#     return total_loss / total, total_correct / total


# @torch.no_grad()
# def evaluate(model, loader, criterion):
#     model.eval()
#     total_loss, total_correct, total = 0.0, 0, 0
#     for xb, yb in loader:
#         xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#         logits = model(xb)
#         loss = criterion(logits, yb)
#         total_loss += loss.item() * xb.size(0)
#         total_correct += (logits.argmax(1) == yb).sum().item()
#         total += xb.size(0)
#     return total_loss / total, total_correct / total


# # ------------------------------------------------------------
# # 5️⃣  Load data + build model
# # ------------------------------------------------------------
# train_ds, test_ds, NUM_CLASSES, FEAT_DIM, SEQ_LEN, INV_LABEL_MAP = load_ts_data(
#     ARGS.csv, ARGS.target, ARGS.window, ARGS.stride, ARGS.test_size
# )
# trainloader = DataLoader(train_ds, batch_size=ARGS.batch, shuffle=True)
# testloader = DataLoader(test_ds, batch_size=ARGS.batch, shuffle=False)

# net = CNN1DClassifier(in_channels=FEAT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=ARGS.lr)

# print(f"[Info] Features: {FEAT_DIM}, SeqLen: {SEQ_LEN}, Classes: {NUM_CLASSES}")
# print(f"[Info] Labels: {INV_LABEL_MAP}")


# # ------------------------------------------------------------
# # 6️⃣  Flower client definition
# # ------------------------------------------------------------
# # AFTER
# class FlowerClient(NumPyClient):
#     def get_parameters(self, config):
#         return [v.cpu().numpy() for _, v in net.state_dict().items()]

#     def set_parameters(self, parameters):
#         params_dict = zip(net.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         net.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         for _ in range(ARGS.epochs_per_fit):
#             loss, acc = train_epoch(net, trainloader, optimizer, criterion)
#         return self.get_parameters({}), len(trainloader.dataset), {"train_loss": loss, "train_acc": acc}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         val_loss, val_acc = evaluate(net, testloader, criterion)
#         return float(val_loss), len(testloader.dataset), {"accuracy": float(val_acc)}


# # ------------------------------------------------------------
# # 7️⃣  Start the Flower client
# # ------------------------------------------------------------
# if __name__ == "__main__":
#     start_utc = datetime.utcnow()
#     print(f"[UTC START] {start_utc.isoformat()} Z")

#     start_client(server_address=ARGS.server, client=FlowerClient().to_client())

#     end_utc = datetime.utcnow()
#     print(f"[UTC END]   {end_utc.isoformat()} Z")
#     print(f"[DURATION]  {end_utc - start_utc}")

# flwr_time_series_client.py — unified CSV/JSON loader for Unity → Flower

from datetime import datetime
from collections import OrderedDict
import argparse, os, json, sys
from typing import Tuple, Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from flwr.client import NumPyClient, start_client

# ------------------------------------------------------------
# 1️⃣  Command-line arguments
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Flower client for Unity CSV/JSON time-series")
    # Allow either --stdin OR --path
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--stdin", action="store_true", help="Read JSON payload from STDIN")
    g.add_argument("--path", help="Path to Unity-exported CSV or JSON")

    p.add_argument("--server", default="127.0.0.1:5006", help="Flower server host:port")
    p.add_argument("--target", default="SceneName", help="Target label column name")
    p.add_argument("--timecol", default=None, help="Optional timestamp column to sort by")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--window", type=int, default=120)
    p.add_argument("--stride", type=int, default=60)
    p.add_argument("--epochs_per_fit", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test_size", type=float, default=0.2)
    return p.parse_args()

ARGS = parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# 2️⃣  Data loading (CSV or JSON) + windowing
# ------------------------------------------------------------
def _guess_frame_list(obj: Any) -> List[dict]:
    """Return the list of frame dicts from a Unity JSON structure.

    Handles common shapes like:
      - top-level list:            [ {...}, {...}, ... ]
      - object with frames key:    { "frames": [ ... ] }
      - object with allFrames key: { "allFrames": [ ... ] }
      - object with data/samples:  { "data": [ ... ] } / { "samples": [ ... ] }
      - single dict (wrap as one)
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["frames", "allFrames", "data", "samples", "records", "items"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        # try one nested level
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return [obj]
    raise ValueError("Could not locate a list of frame objects in the JSON.")

def load_table_from_obj(obj: Any, timecol: Optional[str]) -> pd.DataFrame:
    """Flatten an already-parsed JSON object into a DataFrame (like load_table would)."""
    frames = _guess_frame_list(obj)
    df = pd.json_normalize(frames, sep=".")
    # Optional chronological sort
    if timecol and timecol in df.columns:
        df = df.sort_values(by=timecol)
    # If user didn't specify timecol, try common names
    if not timecol:
        ts_candidates = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
        if ts_candidates:
            df = df.sort_values(by=ts_candidates[0])
    return df

def load_table(path: str, timecol: Optional[str]) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)  # standard JSON
                return load_table_from_obj(obj, timecol)
            except json.JSONDecodeError:
                # Fallback to JSON Lines
                f.seek(0)
                df = pd.read_json(f, lines=True)
                if not isinstance(df, pd.DataFrame):
                    raise
                df = pd.json_normalize(df.to_dict(orient="records"))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Optional chronological sort
    if timecol and timecol in df.columns:
        df = df.sort_values(by=timecol)
    if not timecol:
        ts_candidates = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
        if ts_candidates:
            df = df.sort_values(by=ts_candidates[0])
    return df

def df_to_windows(df: pd.DataFrame, target_col: str, window: int, stride: int):
    assert target_col in df.columns, f"'{target_col}' not found in table columns: {list(df.columns)[:10]}..."

    # Keep only numeric features except the target
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("No numeric feature columns found besides the target.")

    X_all = df[num_cols].to_numpy(dtype=np.float32)

    # Encode labels
    labels, uniques = pd.factorize(df[target_col], sort=True)
    y_all = labels.astype(np.int64)
    inv_label_map = {i: cls for i, cls in enumerate(uniques.tolist())}

    n = len(df)
    X_windows, y_windows = [], []
    i = 0
    while i + window <= n:
        xw = X_all[i:i + window]
        yw_slice = y_all[i:i + window]
        # majority label in the window
        vals, counts = np.unique(yw_slice, return_counts=True)
        y_win = int(vals[np.argmax(counts)]) if len(vals) else int(y_all[min(i + window - 1, n - 1)])
        X_windows.append(xw)
        y_windows.append(y_win)
        i += stride

    X = np.stack(X_windows, axis=0)
    y = np.array(y_windows, dtype=np.int64)

    print(f"[Data] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[Data] Features per sample: {X.shape[2]}, window: {window}, stride: {stride}")
    print(f"[Data] Memory: {X.nbytes/1024**2:.2f} MB")

    return X, y, num_cols, inv_label_map

# For finding out correct column of the data we need this function
def _ensure_column_ci(df: pd.DataFrame, want: str) -> str:
    """Return an existing column name matching `want` case-insensitively.
    If not found, create a case-normalized alias when a close match exists.
    """
    want_l = want.lower()
    mapping = {c.lower(): c for c in df.columns}
    if want_l in mapping:
        return mapping[want_l]
    # common aliases
    aliases = {
        "scenename": ["scene", "scene_name", "scenelabel", "label"],
        "elapsedtime": ["time", "timestamp", "t", "elapsed_time"],
    }
    for cand in aliases.get(want_l, []):
        if cand in mapping:
            real = mapping[cand]
            df[want] = df[real]
            return want
    raise KeyError(f"Column '{want}' not found (available: {list(df.columns)[:12]} …)")

def load_ts_data_from_obj(obj: Any, target_col: str, window: int, stride: int, test_size: float, timecol: Optional[str]):
    df = load_table_from_obj(obj, timecol=timecol)
    target_col = _ensure_column_ci(df, target_col)
    if timecol:
        timecol = _ensure_column_ci(df, timecol)

    X, y, feat_cols, inv_label_map = df_to_windows(df, target_col, window, stride)
    stratify_arg = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True, stratify=stratify_arg
    )

    # [B, C, T] for Conv1D
    X_train_t = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).transpose(1, 2)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    num_classes = int(y.max()) + 1 if len(y) else 1
    feat_dim = X_train_t.shape[1] if len(X_train_t) else (X_test_t.shape[1] if len(X_test_t) else 0)
    seq_len  = X_train_t.shape[2] if len(X_train_t) else (X_test_t.shape[2] if len(X_test_t) else ARGS.window)

    train_ds = TensorDataset(X_train_t, y_train_t) if len(X_train_t) else TensorDataset(torch.empty(0, feat_dim, seq_len), torch.empty(0, dtype=torch.long))
    test_ds  = TensorDataset(X_test_t,  y_test_t)  if len(X_test_t)  else TensorDataset(torch.empty(0, feat_dim, seq_len), torch.empty(0, dtype=torch.long))
    return train_ds, test_ds, num_classes, feat_dim, seq_len, inv_label_map

def load_ts_data(path: str, target_col: str, window: int, stride: int, test_size: float, timecol: Optional[str]):
    df = load_table(path, timecol=timecol)
    target_col = _ensure_column_ci(df, target_col)
    if timecol:
        timecol = _ensure_column_ci(df, timecol)

    X, y, feat_cols, inv_label_map = df_to_windows(df, target_col, window, stride)
    stratify_arg = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True, stratify=stratify_arg
    )

    # [B, C, T] for Conv1D
    X_train_t = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).transpose(1, 2)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    num_classes = int(y.max()) + 1
    feat_dim = X_train_t.shape[1]
    seq_len = X_train_t.shape[2]

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds  = TensorDataset(X_test_t,  y_test_t)
    return train_ds, test_ds, num_classes, feat_dim, seq_len, inv_label_map


# ------------------------------------------------------------
# 3️⃣  Model
# ------------------------------------------------------------
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
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.head(x)


# ------------------------------------------------------------
# 4️⃣  Train / eval helpers
# ------------------------------------------------------------
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
        total_correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / max(total,1), total_correct / max(total,1)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / max(total,1), total_correct / max(total,1)


# ------------------------------------------------------------
# 5️⃣  Load data + build model
# ------------------------------------------------------------
if ARGS.stdin:
    raw = sys.stdin.read()
    if not raw:
        raise ValueError("STDIN is empty; expected JSON data")
    try:
        json_obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from STDIN: {e}")
    train_ds, test_ds, NUM_CLASSES, FEAT_DIM, SEQ_LEN, INV_LABEL_MAP = load_ts_data_from_obj(
        json_obj, ARGS.target, ARGS.window, ARGS.stride, ARGS.test_size, ARGS.timecol
    )
else:
    train_ds, test_ds, NUM_CLASSES, FEAT_DIM, SEQ_LEN, INV_LABEL_MAP = load_ts_data(
        ARGS.path, ARGS.target, ARGS.window, ARGS.stride, ARGS.test_size, ARGS.timecol
    )

trainloader = DataLoader(train_ds, batch_size=ARGS.batch, shuffle=True)
testloader  = DataLoader(test_ds,  batch_size=ARGS.batch, shuffle=False)

net = CNN1DClassifier(in_channels=FEAT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=ARGS.lr)

print(f"[Info] Features: {FEAT_DIM}, SeqLen: {SEQ_LEN}, Classes: {NUM_CLASSES}")
print(f"[Info] Labels: {INV_LABEL_MAP}")


# ------------------------------------------------------------
# 6️⃣  Flower client definition
# ------------------------------------------------------------
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [v.detach().cpu().numpy() for _, v in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = 0.0, 0.0
        for _ in range(ARGS.epochs_per_fit):
            loss, acc = train_epoch(net, trainloader, optimizer, criterion)
        return self.get_parameters({}), len(trainloader.dataset), {"train_loss": float(loss), "train_acc": float(acc)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, val_acc = evaluate(net, testloader, criterion)
        return float(val_loss), len(testloader.dataset), {"accuracy": float(val_acc)}


# ------------------------------------------------------------
# 7️⃣  Start Flower client
# ------------------------------------------------------------
if __name__ == "__main__":
    start_utc = datetime.utcnow()
    print(f"[UTC START] {start_utc.isoformat()} Z")

    start_client(server_address=ARGS.server, client=FlowerClient().to_client())

    end_utc = datetime.utcnow()
    print(f"[UTC END]   {end_utc.isoformat()} Z")
    print(f"[DURATION]  {end_utc - start_utc}")
