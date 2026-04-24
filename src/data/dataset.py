import torch
from torch.utils.data import Dataset


class BonnDataset(Dataset):

    def __init__(self, data, labels, train=False):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.train = train

        if self.data.ndim != 4:
            raise ValueError(
                f"Expected data shape (N, C, H, W), got {self.data.shape}"
            )

    def __len__(self):
        return self.data.shape[0]

    # -----------------------------
    # AUGMENTATION
    # -----------------------------
    def augment(self, x):

        # gaussian noise
        if torch.rand(()) < 0.3:
            noise_level = 0.005 + 0.01 * torch.rand(())
            x = x + noise_level * torch.randn_like(x)

        # amplitude scaling
        if torch.rand(()) < 0.3:
            scale = 0.9 + 0.2 * torch.rand(())
            x = x * scale

        # frequency masking
        if torch.rand(()) < 0.3:
            f = int(torch.randint(1, x.shape[2]//4 + 1, ()).item())
            max_start = max(1, x.shape[2] - f)
            start = int(torch.randint(0, max_start, ()).item())
            x[:, :, start:start+f] = 0

        return x

    # -----------------------------
    # NORMALIZATION
    # -----------------------------
    def normalize(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        std = torch.clamp(std, min=1e-5)
        return (x - mean) / std

    def __getitem__(self, idx):

        x = self.data[idx].clone()
        y = self.labels[idx]

        if self.train:
            x = self.augment(x)

        x = self.normalize(x)

        return x, y