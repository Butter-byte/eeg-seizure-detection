import torch
from torch.utils.data import Dataset


class BonnDataset(Dataset):
    """
    PyTorch Dataset for Bonn EEG CWT scalograms.

    FIXES:
        - augment() now clamps output to [-6, 6] to prevent extreme values
          leaking into the normalisation step in train.py
        - Frequency / time masking max-fraction raised slightly (1/5 vs 1/6)
          for better regularisation without destroying signal
        - clone() called before augment to avoid mutating the cached tensor

    ACCURACY IMPROVEMENTS:
        - Mixup removed (was never in this version, left clean)
        - Augmentation probabilities left as-is (well-tuned)
        - No double normalisation: train.py handles per-batch norm (correct)
    """

    def __init__(self, data, labels, train=False):
        self.data   = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.train  = train

        if self.data.ndim != 4:
            raise ValueError(
                f"Expected data shape (N, C, H, W), got {self.data.shape}"
            )

    def __len__(self):
        return self.data.shape[0]

    # ------------------------------------------------------------------
    # AUGMENTATION
    # ------------------------------------------------------------------
    def augment(self, x):
        """x: (C, H, W) float tensor"""

        # ── Gaussian noise ────────────────────────────────────────────
        if torch.rand(()) < 0.3:
            noise_level = 0.003 + 0.007 * torch.rand(())
            x = x + noise_level * torch.randn_like(x)

        # ── Amplitude scaling ─────────────────────────────────────────
        if torch.rand(()) < 0.3:
            scale = 0.9 + 0.2 * torch.rand(())
            x = x * scale

        # ── SpecAugment: frequency masking (rows = time axis) ─────────
        if torch.rand(()) < 0.4:
            H     = x.shape[1]
            max_f = max(1, H // 5)                          # FIX: 1/5 (was 1/6)
            f     = int(torch.randint(1, max_f + 1, ()).item())
            start = int(torch.randint(0, max(1, H - f), ()).item())
            x = x.clone()
            x[:, start:start + f, :] = 0.0

        # ── SpecAugment: time masking (cols = frequency axis) ─────────
        if torch.rand(()) < 0.4:
            W     = x.shape[2]
            max_t = max(1, W // 5)                          # FIX: 1/5 (was 1/6)
            t     = int(torch.randint(1, max_t + 1, ()).item())
            start = int(torch.randint(0, max(1, W - t), ()).item())
            x = x.clone()
            x[:, :, start:start + t] = 0.0

        # FIX: clamp to prevent extreme values before batch normalisation
        x = torch.clamp(x, -6.0, 6.0)

        return x

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        x = self.data[idx].clone()   # always clone — safe for augmentation
        y = self.labels[idx]

        if self.train:
            x = self.augment(x)

        return x, y
