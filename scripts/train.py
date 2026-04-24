import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.models.maven_net import MavenNet
from src.data.dataset import BonnDataset
from src.data.cwt import generate_cwt
from preprocessing_pipeline import load_bonn_raw
from src.utils.metrics import compute_metrics


# -----------------------------
# SEED
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


# -----------------------------
# EXPERIMENTS
# -----------------------------
EXPERIMENTS = [
    {"name": "ABCD_vs_E", "normal": [0,1,2,3]},
    {"name": "A_vs_E",    "normal": [0]},
    {"name": "B_vs_E",    "normal": [1]},
    {"name": "C_vs_E",    "normal": [2]},
    {"name": "D_vs_E",    "normal": [3]},
    {"name": "AB_vs_E",   "normal": [0,1]},
    {"name": "CD_vs_E",   "normal": [2,3]},
]

SEIZURE_LABEL = 4


# -----------------------------
# FILTER
# -----------------------------
def filter_experiment(X, y, groups, normal_classes):
    mask = np.isin(y, normal_classes + [SEIZURE_LABEL])
    X = X[mask]
    y = y[mask]
    g = groups[mask]

    y = (y == SEIZURE_LABEL).astype(np.int64)
    return X, y, g


# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize_batch(x):
    mean = x.mean(dim=(2,3), keepdim=True)
    std = x.std(dim=(2,3), keepdim=True) + 1e-8
    return (x - mean) / std


# -----------------------------
# SINGLE FOLD
# -----------------------------
def run_fold(X_train_full, y_train_full, g_train_full,
             X_test, y_test, device, fold, exp_name):

    gss = GroupShuffleSplit(test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(X_train_full, y_train_full, groups=g_train_full))

    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

    if len(X_val) == 0 or len(np.unique(y_train)) < 2:
        return None

    # -----------------------------
    # SAMPLER (FIXED)
    # -----------------------------
    class_counts = np.bincount(y_train, minlength=2)
    weights = 1.0 / (class_counts + 1e-6)
    sample_weights = weights[y_train]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    def loader(X, y, shuffle=False, sampler=None):
        return DataLoader(
            BonnDataset(X, y),
            batch_size=64,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=0,
            pin_memory=(device.type == "cuda")
        )

    train_loader = loader(X_train, y_train, sampler=sampler)
    val_loader   = loader(X_val, y_val)
    test_loader  = loader(X_test, y_test)

    model = MavenNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_loss = float("inf")
    patience = 10
    counter = 0

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{exp_name}_fold_{fold}.pth"

    # -----------------------------
    # TRAIN
    # -----------------------------
    for epoch in range(50):

        model.train()
        train_loss = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = normalize_batch(x)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = criterion(model(x), y)

            if torch.isnan(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * x.size(0)
            total += x.size(0)

        if total == 0:
            return None

        train_loss /= total

        # VALIDATION
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = normalize_batch(x)
                val_loss += criterion(model(x), y).item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"{exp_name} | Fold {fold} | Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            counter += 1
            if counter >= patience:
                break

    # -----------------------------
    # TEST
    # -----------------------------
    if not os.path.exists(ckpt_path):
        return None

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    probs, labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            x = normalize_batch(x)
            p = torch.softmax(model(x), dim=1)[:,1]
            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    return compute_metrics(labels, probs)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    data, labels, groups = load_bonn_raw("data/raw")

    os.makedirs("data/processed", exist_ok=True)
    path = "data/processed/cwt_data.npy"

    if not os.path.exists(path):
        cwt_data = np.array([generate_cwt(x) for x in data], dtype=np.float32)
        np.save(path, cwt_data)
    else:
        cwt_data = np.load(path).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []

    for exp in EXPERIMENTS:

        print(f"\nRunning: {exp['name']}")

        X, y, g = filter_experiment(cwt_data, labels, groups, exp["normal"])

        gkf = GroupKFold(n_splits=5)
        results = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=g), 1):

            metrics = run_fold(
                X[train_idx], y[train_idx], g[train_idx],
                X[test_idx], y[test_idx],
                device, fold, exp["name"]
            )

            if metrics is not None:
                results.append(metrics)

        if len(results) == 0:
            continue

        final = {
            k: (np.mean([r[k] for r in results]),
                np.std([r[k] for r in results]))
            for k in results[0]
        }

        all_results.append((exp["name"], final))

    os.makedirs("outputs", exist_ok=True)

    rows = []
    for name, r in all_results:
        rows.append({
            "Experiment": name,
            **{f"{k}_mean": v[0] for k,v in r.items()},
            **{f"{k}_std": v[1] for k,v in r.items()}
        })

    pd.DataFrame(rows).to_csv("outputs/results.csv", index=False)

    print("Training complete.")