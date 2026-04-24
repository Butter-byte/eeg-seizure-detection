import os
import glob
import numpy as np


def load_bonn_raw(root_dir="data/raw"):
    """
    Load Bonn EEG dataset from TXT files.

    Returns:
        X      : (N, 178)
        y      : (N,) labels {0,1,2,3,4}
        groups : (N,) file-level grouping (int IDs)
    """

    X_list = []
    y_list = []
    group_list = []

    label_map = {
        "Z": 0,
        "O": 1,
        "N": 2,
        "F": 3,
        "S": 4
    }

    group_id = 0  # ✅ compact group indexing

    for set_name, label in label_map.items():

        set_path = os.path.join(root_dir, set_name)

        if not os.path.exists(set_path):
            print(f"Warning: {set_path} not found. Skipping...")
            continue

        files = glob.glob(os.path.join(set_path, "**", "*.txt"), recursive=True)

        if len(files) == 0:
            print(f"Warning: No TXT files found in {set_path}")
            continue

        print(f"Loading {len(files)} files from {set_name}...")

        for fpath in sorted(files):

            try:
                signal = np.loadtxt(fpath)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")
                continue

            # -----------------------------
            # SANITY CHECKS
            # -----------------------------
            if signal.ndim != 1:
                continue

            if len(signal) < 178:
                continue

            if not np.isfinite(signal).all():
                continue

            if np.std(signal) < 1e-6:
                continue

            # -----------------------------
            # DYNAMIC CHUNKING (FIXED)
            # -----------------------------
            n_chunks = len(signal) // 178

            for i in range(n_chunks):
                start = i * 178
                end = start + 178

                chunk = signal[start:end]

                if len(chunk) != 178:
                    continue

                X_list.append(chunk)
                y_list.append(label)
                group_list.append(group_id)

            group_id += 1  # increment per file

    if len(X_list) == 0:
        raise ValueError("No data loaded. Check dataset path or file format.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(group_list, dtype=np.int64)

    print(f"Total samples loaded: {X.shape[0]}")
    print(f"Unique groups: {len(np.unique(groups))}")

    # Keep raw signals (CWT handles normalization)
    X = np.ascontiguousarray(X)

    return X, y, groups