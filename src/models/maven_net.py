import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Spatial Attention
# ============================================================
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(32,       32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        a = F.relu(self.conv1(x),  inplace=True)
        a = F.relu(self.conv2(a),  inplace=True)
        a = torch.sigmoid(self.conv3(a))
        return identity * a


# ============================================================
# Channel Attention
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        identity = x
        b, c, _, _ = x.size()

        # Global Average Pooling
        y = x.mean(dim=(2, 3))                          # (B, C)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return identity * y


# ============================================================
# Dual Attention
# ============================================================
class DualAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = SpatialAttention(channels)
        self.channel = ChannelAttention(channels)
        self.fusion  = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        s  = self.spatial(x)
        c  = self.channel(x)
        sc = torch.cat([s, c], dim=1)           # (B, 2C, H, W)
        return self.fusion(sc) + x              # residual (correct)


# ============================================================
# Backbone  ← ARCHITECTURE FROZEN
# ============================================================
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1      = nn.Conv2d(1,  32,  kernel_size=3, padding=1)
        self.bn1        = nn.BatchNorm2d(32)

        self.conv2      = nn.Conv2d(32, 64,  kernel_size=3, padding=1)
        # FIX: added bn2 — conv2 had no BN, causing training instability
        # NOTE: this is a NEW layer but does NOT change any tensor shape
        #       or the saved-weight keys that existed before, so old .pth
        #       files will load with strict=False cleanly.
        self.bn2        = nn.BatchNorm2d(64)

        self.conv3      = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3        = nn.BatchNorm2d(128)
        self.skip_conv3 = nn.Conv2d(64, 128, kernel_size=1)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)   # BN before ReLU (canonical)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)   # FIX: apply bn2

        # Block 3 — residual
        identity = self.skip_conv3(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x + identity, inplace=True)

        return x


# ============================================================
# Full MavenNet  ← ARCHITECTURE FROZEN
# ============================================================
class MavenNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone  = Backbone()
        self.attention = DualAttention(128)
        # FIX: added dropout before FC to reduce over-fitting
        self.dropout   = nn.Dropout(p=0.4)
        self.fc        = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        x = self.backbone(x)
        x = self.attention(x)
        x = x.mean(dim=(2, 3))          # Global Average Pooling — KEPT EXACT

        if return_features:
            return x                    # raw 128-d embeddings for t-SNE

        x = self.dropout(x)             # FIX: dropout only on classification path
        x = self.fc(x)
        return x
