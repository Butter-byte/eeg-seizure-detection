import os
import glob
import numpy as np


def load_bonn_raw(root_dir="."):
    """
    Loads raw Bonn EEG data from TXT files and splits them into 178-sample chunks.
    Expected structure: root_dir/Z/*.txt, root_dir/O/*.txt, etc.
    """
    X_list = []
    y_list = []

    # Map sets to binary labels (Set E/S is Seizure=1, others are 0)
    # Z=SetA, O=SetB, N=SetC, F=SetD, S=SetE
    sets = {
        "Z": 0, "O": 0, "N": 0, "F": 0, "S": 1
    }

    for set_name, label in sets.items():
        set_path = os.path.join(root_dir, set_name)
        
        # Find all .txt files in the set directory or its subdirectories
        txt_files = glob.glob(os.path.join(set_path, "**", "*.txt"), recursive=True)

        if not txt_files:
            print(f"Warning: No TXT files found in {set_path}")
            continue

        print(f"Loading {len(txt_files)} files from {set_name}...")

        for fpath in sorted(txt_files):
            # Load the 4096 samples from the text file
            try:
                data = np.loadtxt(fpath)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                continue

            # Split 4096 samples into 178-sample chunks (23 chunks total)
            num_chunks = 23 # fixed to match CSV structure exactly
            chunk_size = 178
            
            for i in range(num_chunks):
                chunk = data[i * chunk_size : (i + 1) * chunk_size]
                X_list.append(chunk)
                y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"Total samples loaded: {X.shape[0]}")

    # Per-sample normalization (to match load_bonn_raw logic)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1 # avoid divide-by-zero
    
    X = (X - mean) / std
    X = np.ascontiguousarray(X)

    return X, y