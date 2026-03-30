import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# Spatial Attention
# ============================================
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        # Match latest diagram: 3-layer bottleneck (3x3 -> 3x3 -> 1x1)
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return identity * x


# ============================================
# Channel Attention
# ============================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        # Match diagram: ReduceMean -> Linear -> ReLU -> Linear -> Sigmoid -> Multiply
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        identity = x
        b, c, _, _ = x.size()
        
        # ReduceMean (global avg pool)
        y = torch.mean(x, dim=(2, 3)) 
        
        # Gemm + ReLU + Gemm + Sigmoid
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        
        # Reshape to match spatial dims for Mul
        y = y.view(b, c, 1, 1)
        return identity * y


# ============================================
# Dual Attention
# ============================================
class DualAttention(nn.Module):
    def __init__(self, channels):
        super(DualAttention, self).__init__()
        self.spatial = SpatialAttention(channels)
        self.channel = ChannelAttention(channels)
        
        # Fusing the concatenated spatial and channel results
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        # Diagram: Concatenate branches and then Residual Add with block input
        s = self.spatial(x)
        c = self.channel(x)
        
        sc = torch.cat([s, c], dim=1) # Concat (b=256)
        fused = self.fusion(sc) # Fusion 1x1 Conv (b=128)
        
        return fused + x # Final Residual Add


# ============================================
# Backbone
# ============================================
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        # Block 1: Conv 32 + BN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: Conv 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Block 3: Residual Block 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.skip_conv3 = nn.Conv2d(64, 128, kernel_size=1) 

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = self.bn1(x) 

        # Block 2
        x = F.relu(self.conv2(x))

        # Block 3: Residual Block (Add -> Relu sequence in diagram)
        identity = self.skip_conv3(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(x + identity) # Added Relu after the sum as per final diagram

        return x


# ============================================
# Full MavenNet
# ============================================
class MavenNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MavenNet, self).__init__()

        self.backbone = Backbone()
        self.attention = DualAttention(128)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.backbone(x)
        x = self.attention(x)

        # Global Average Pooling (ReduceMean)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)

        return x