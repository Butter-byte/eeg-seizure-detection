import torch
from torch.utils.data import Dataset


class BonnDataset(Dataset):

    def __init__(self, data, labels):
        """
        data: numpy array (N, 1, H, W)
        labels: numpy array (N,)
        """

        # Convert once, not per sample
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Optional sanity check
        if self.data.ndim != 4:
            raise ValueError(
                f"Expected data shape (N, C, H, W), got {self.data.shape}"
            )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]