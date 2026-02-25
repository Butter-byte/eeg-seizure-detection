import pandas as pd
import numpy as np


def load_bonn_csv(csv_path):

    df = pd.read_csv(csv_path)

    # Remove any unnamed columns safely
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Features
    X = df.drop(columns=["y"]).values.astype(np.float32)

    # Raw labels
    y_raw = df["y"].values.astype(int)

    # Binary classification:
    # 1 = seizure
    # 2–5 = non-seizure
    y = (y_raw == 1).astype(np.int64)

    # Per-sample normalization (no leakage)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)

    std[std == 0] = 1  # avoid divide-by-zero

    X = (X - mean) / std

    X = np.ascontiguousarray(X)

    return X, y