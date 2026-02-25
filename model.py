import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# Spatial Attention
# ============================================
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()

        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        s = F.relu(self.conv3x3(x))
        s = torch.sigmoid(self.conv1x1(s))
        return x * s


# ============================================
# Channel Attention
# ============================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()

        c_attn = self.gap(x).view(b, c)
        c_attn = F.relu(self.fc1(c_attn))
        c_attn = torch.sigmoid(self.fc2(c_attn))
        c_attn = c_attn.view(b, c, 1, 1)

        return x * c_attn


# ============================================
# Dual Attention
# ============================================
class DualAttention(nn.Module):
    def __init__(self, channels):
        super(DualAttention, self).__init__()

        self.spatial = SpatialAttention(channels)
        self.channel = ChannelAttention(channels)

        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        s = self.spatial(x)
        c = self.channel(x)

        concat = torch.cat([s, c], dim=1)
        fused = self.fusion(concat)

        return fused + x


# ============================================
# Backbone (UPDATED AS YOU REQUESTED)
# ============================================
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        # Conv1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn_after_relu1 = nn.BatchNorm2d(32)  # NEW BN after first ReLU

        # Conv2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Conv3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_after_relu3 = nn.BatchNorm2d(128)  # NEW BN before branching

        # Skip projection (no pooling now, same spatial size)
        self.skip_conv = nn.Conv2d(64, 128, kernel_size=1)

    def forward(self, x):

        # -------- Block 1 --------
        x = F.relu(self.conv1(x))
        x = self.bn_after_relu1(x)  # BN after first ReLU

        # -------- Block 2 --------
        x = F.relu(self.conv2(x))

        skip = self.skip_conv(x)

        # -------- Block 3 --------
        x = F.relu(self.conv3(x))
        x = self.bn_after_relu3(x)  # BN before branching

        # Residual Add
        x = x + skip

        return x


# ============================================
# Full MavenNet
# ============================================
class MavenNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MavenNet, self).__init__()

        self.backbone = Backbone()
        self.attention = DualAttention(128)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.backbone(x)
        x = self.attention(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x