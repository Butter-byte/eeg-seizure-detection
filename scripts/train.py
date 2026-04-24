import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import subprocess

# -----------------------------
# Fix imports
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, train_test_split

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

set_seed()


# -----------------------------
# EXPERIMENTS
# -----------------------------
EXPERIMENTS = [
    {"name": "ABCD vs E", "normal": [0,1,2,3]},
    {"name": "A vs E",    "normal": [0]},
    {"name": "B vs E",    "normal": [1]},
    {"name": "C vs E",    "normal": [2]},
    {"name": "D vs E",    "normal": [3]},
    {"name": "AB vs E",   "normal": [0,1]},
    {"name": "CD vs E",   "normal": [2,3]},
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
# SINGLE FOLD
# -----------------------------
def run_fold(X_train_full, y_train_full, X_test, y_test, device, fold):

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.1,
        stratify=y_train_full,
        random_state=42
    )

    def loader(X, y, shuffle=False):
        return DataLoader(
            BonnDataset(X, y),
            batch_size=64,
            shuffle=shuffle,
            num_workers=0,        # ✅ Mac safe
            pin_memory=False      # ✅ no warning
        )

    train_loader = loader(X_train, y_train, True)
    val_loader   = loader(X_val, y_val)
    test_loader  = loader(X_test, y_test)

    model = MavenNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    # Class weights
    counts = np.bincount(y_train, minlength=2)
    weights = torch.tensor([
        len(y_train)/(counts[0]+1e-8),
        len(y_train)/(counts[1]+1e-8)
    ], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    best_loss = float("inf")
    patience = 7
    counter = 0

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/fold_{fold}.pth"

    # -----------------------------
    # TRAIN
    # -----------------------------
    for epoch in range(25):   # balanced speed vs accuracy

        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Fold {fold} | Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    # -----------------------------
    # TEST
    # -----------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    probs, labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            p = torch.softmax(model(x), dim=1)[:,1]
            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    return compute_metrics(labels, probs)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("Loading dataset...")
    data, labels, groups = load_bonn_raw("data/raw")

    # -----------------------------
    # CWT CACHE
    # -----------------------------
    os.makedirs("data/processed", exist_ok=True)
    path = "data/processed/cwt_data.npy"

    if not os.path.exists(path):
        print("Generating CWT...")
        cwt_data = np.array([generate_cwt(x) for x in data], dtype=np.float32)
        np.save(path, cwt_data)
    else:
        print("Loading cached CWT...")
        cwt_data = np.load(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    all_results = []

    # -----------------------------
    # EXPERIMENT LOOP
    # -----------------------------
    for exp in EXPERIMENTS:

        print(f"\nRunning: {exp['name']}")

        X, y, g = filter_experiment(cwt_data, labels, groups, exp["normal"])

        gkf = GroupKFold(n_splits=5)
        results = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=g), 1):

            # HARD LEAK CHECK
            assert len(set(g[train_idx]) & set(g[test_idx])) == 0

            metrics = run_fold(
                X[train_idx], y[train_idx],
                X[test_idx], y[test_idx],
                device, fold
            )

            results.append(metrics)

        final = {
            k: (np.mean([r[k] for r in results]),
                np.std([r[k] for r in results]))
            for k in results[0]
        }

        all_results.append((exp["name"], final))

    # -----------------------------
    # SAVE RESULTS
    # -----------------------------
    os.makedirs("outputs", exist_ok=True)

    rows = []
    print("\n" + "="*90)

    for name, r in all_results:
        print(f"{name:<15} | Acc {r['accuracy'][0]:.3f} | F1 {r['f1'][0]:.3f}")

        rows.append({
            "Experiment": name,
            **{f"{k}_mean": v[0] for k,v in r.items()},
            **{f"{k}_std": v[1] for k,v in r.items()}
        })

    print("="*90)

    pd.DataFrame(rows).to_csv("outputs/results.csv", index=False)
    print("Saved → outputs/results.csv")

    # -----------------------------
    # AUTO PIPELINE
    # -----------------------------
    print("\nRunning pipeline...")
    result = subprocess.run(
    [sys.executable, "scripts/pipeline.py"],
    capture_output=True,
    text=True
    )
    
    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Pipeline FAILED")