import os, glob, json
import numpy as np
import pandas as pd
import torch
import flwr as fl
from torch.utils.data import DataLoader, TensorDataset
from tcn_model import TCNModel

# --- PATHS ---
SESS_DIR   = r"C:\XRFL\unity_sessions"
MODEL_DIR  = r"C:\XRFL\models"
STATE_FILE = r"C:\XRFL\code\trainer_state.json"
SERVER     = "127.0.0.1:8080"   # CHANGE to your FL server IP if remote

# --- MODEL / TRAIN ---
C       = 6        # features: e.g., LeftDirX,LeftDirY,LeftDirZ,RightDirX,RightDirY,RightDirZ
T       = 64       # window length
STRIDE  = 16
BATCH   = 64
LR      = 1e-3
EPOCHS  = 2
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_FEATURE_COLS = ["LeftDirX","LeftDirY","LeftDirZ","RightDirX","RightDirY","RightDirZ"]
DEFAULT_LABEL_COL    = "Label"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"processed_sessions": []}

def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)

def list_new_sessions(processed):
    sess = sorted([d for d in glob.glob(os.path.join(SESS_DIR, "*")) if os.path.isdir(d)])
    return [s for s in sess if s not in processed and os.path.exists(os.path.join(s,"session.csv"))]

def read_manifest(sess_dir):
    manp = os.path.join(sess_dir, "manifest.json")
    if os.path.exists(manp):
        try:
            with open(manp, "r") as f:
                man = json.load(f)
            feats = man.get("feature_cols", DEFAULT_FEATURE_COLS)
            label = man.get("label_col", DEFAULT_LABEL_COL)
            return feats, label
        except Exception:
            pass
    return DEFAULT_FEATURE_COLS, DEFAULT_LABEL_COL

def make_windows(X, y, T=64, stride=16):
    # X: [N,C], y: [N]
    xs, ys = [], []
    for start in range(0, len(X) - T + 1, stride):
        xs.append(X[start:start+T].T)  # [C,T]
        ys.append(np.bincount(y[start:start+T]).argmax())  # majority label for the window
    return np.stack(xs), np.array(ys)

def load_sessions(new_sessions):
    if not new_sessions:
        return None

    frames = []
    feat_cols, label_col = None, None
    # Merge sessions (you can choose per-session training instead if you prefer)
    for s in new_sessions:
        csvp = os.path.join(s, "session.csv")
        if not os.path.exists(csvp):
            continue
        df = pd.read_csv(csvp)
        fc, lc = read_manifest(s)
        feat_cols = fc if feat_cols is None else feat_cols
        label_col = lc if label_col is None else label_col
        frames.append(df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    X = df[feat_cols].astype("float32").values
    y = df[label_col].astype("int64").values

    # Per-row L2 norm (optional but common for direction vectors)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
    X = X / n

    Xw, yw = make_windows(X, y, T=T, stride=STRIDE)
    Xw = torch.tensor(Xw, dtype=torch.float32)  # [Nw,C,T]
    yw = torch.tensor(yw, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xw, yw), batch_size=BATCH, shuffle=True)
    return loader, feat_cols, label_col

class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = TCNModel(C=C, T=T, num_classes=3).to(DEVICE)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.state = load_state()

    def get_parameters(self, _config):
        return [v.detach().cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, params):
        sd = self.model.state_dict()
        for (k, _), p in zip(sd.items(), params):
            sd[k] = torch.tensor(p)
        self.model.load_state_dict(sd, strict=True)

    def fit(self, params, config):
        self.set_parameters(params)
        new_sessions = list_new_sessions(self.state["processed_sessions"])
        if not new_sessions:
            # no new local data -> send back unchanged weights
            return self.get_parameters({}), 0, {}

        loader_feats = load_sessions(new_sessions)
        if loader_feats is None:
            return self.get_parameters({}), 0, {}
        loader, feat_cols, label_col = loader_feats

        self.model.train()
        seen = 0
        for _ in range(EPOCHS):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                self.opt.zero_grad()
                logits = self.model(xb)
                loss = self.model.loss_fn(logits, yb)
                loss.backward()
                self.opt.step()
                seen += len(xb)

        # mark these sessions processed
        self.state["processed_sessions"].extend(new_sessions)
        save_state(self.state)

        # (optional) export a local TorchScript for your records
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model.eval()
        example = torch.randn(1, C, T).to(DEVICE)
        traced = torch.jit.trace(self.model, example)
        traced.save(os.path.join(MODEL_DIR, "tcn_eye.torchscript.pt"))

        return self.get_parameters({}), seen, {}

    def evaluate(self, params, config):
        # optional: implement local validation if you have a split
        return 0.0, 0, {}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address=SERVER, client=FLClient())
