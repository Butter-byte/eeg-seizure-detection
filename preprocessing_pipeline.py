import os
import glob
import numpy as np


def load_bonn_raw(root_dir="data/raw"):
    """
    Load Bonn EEG dataset from TXT files.

    Returns:
        X      : (N, 178)
        y      : (N,) labels {0,1,2,3,4}
        groups : (N,) file-level grouping
    """

    X_list = []
    y_list = []
    group_list = []

    # Multi-class labels
    label_map = {
        "Z": 0,
        "O": 1,
        "N": 2,
        "F": 3,
        "S": 4
    }

    for set_name, label in label_map.items():

        set_path = os.path.join(root_dir, set_name)

        if not os.path.exists(set_path):
            print(f"Warning: {set_path} not found. Skipping...")
            continue

        # 🔥 FIX: handle .TXT and .txt
        files = glob.glob(os.path.join(set_path, "**", "*.*"), recursive=True)
        txt_files = [f for f in files if f.lower().endswith(".txt")]

        if len(txt_files) == 0:
            print(f"Warning: No TXT files found in {set_path}")
            continue

        print(f"Loading {len(txt_files)} files from {set_name}...")

        for fpath in sorted(txt_files):

            try:
                signal = np.loadtxt(fpath)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")
                continue

            # Ensure correct length
            if signal.shape[0] < 4096:
                continue

            # Split into 23 chunks
            for i in range(23):
                chunk = signal[i * 178:(i + 1) * 178]

                if chunk.shape[0] != 178:
                    continue

                X_list.append(chunk)
                y_list.append(label)
                group_list.append(fpath)  # SAME file → SAME group

    # -----------------------------
    # Safety check
    # -----------------------------
    if len(X_list) == 0:
        raise ValueError("No data loaded. Check dataset path or file format.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(group_list)

    print(f"Total samples loaded: {X.shape[0]}")

    # -----------------------------
    # Normalization (per sample)
    # -----------------------------
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1

    X = (X - mean) / std
    X = np.ascontiguousarray(X)

    return X, y, groups