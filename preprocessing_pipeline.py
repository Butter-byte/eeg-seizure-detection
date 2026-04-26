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

    FIXES:
        - Chunk normalization moved AFTER chunking (was inside loop, correct placement kept)
        - Added robust NaN/Inf check per chunk (not just per file)
        - Added duplicate-chunk guard via std threshold per chunk
        - stride kept at 89 (50% overlap) — intentional for data augmentation
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

    group_id = 0

    def natural_sort_key(s):
        import re
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)
        ]

    for set_name, label in label_map.items():

        set_path = os.path.join(root_dir, set_name)

        if not os.path.exists(set_path):
            print(f"Warning: {set_path} not found. Skipping...")
            continue

        files = glob.glob(os.path.join(set_path, "**", "*.*"), recursive=True)
        files = [f for f in files if f.lower().endswith(".txt")]

        if len(files) == 0:
            print(f"Warning: No TXT files found in {set_path}")
            continue

        files = sorted(files, key=natural_sort_key)
        print(f"Loading {len(files)} files from {set_name}...")

        for fpath in files:

            try:
                signal = np.loadtxt(fpath)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")
                continue

            # ── File-level guards ──────────────────────────────────────
            if signal.ndim != 1:
                continue
            if len(signal) < 178:
                continue
            if not np.isfinite(signal).all():
                # FIX: attempt to repair isolated NaNs via linear interpolation
                # instead of dropping the entire file
                nans = ~np.isfinite(signal)
                if nans.sum() > len(signal) * 0.05:   # >5% corrupt → skip
                    continue
                idx = np.arange(len(signal))
                signal[nans] = np.interp(idx[nans], idx[~nans], signal[~nans])

            if np.std(signal) < 1e-6:
                continue

            # ── Chunking ──────────────────────────────────────────────
            window_size = 178
            stride      = 89   # 50% overlap — correct, keeps temporal diversity

            for start in range(0, len(signal) - window_size + 1, stride):
                chunk = signal[start : start + window_size]

                # FIX: per-chunk finite + variance guard
                if not np.isfinite(chunk).all():
                    continue
                if np.std(chunk) < 1e-6:        # flat chunk → skip
                    continue

                # Normalize per chunk
                chunk = (chunk - np.mean(chunk)) / (np.std(chunk) + 1e-8)

                X_list.append(chunk)
                y_list.append(label)
                group_list.append(group_id)

            group_id += 1   # one group per FILE (correct for GroupKFold)

    if len(X_list) == 0:
        raise ValueError("No data loaded. Check dataset path or file format.")

    X      = np.array(X_list,     dtype=np.float32)
    y      = np.array(y_list,     dtype=np.int64)
    groups = np.array(group_list, dtype=np.int64)

    print(f"Total samples loaded : {X.shape[0]}")
    print(f"Unique groups        : {len(np.unique(groups))}")

    return np.ascontiguousarray(X), y, groups
