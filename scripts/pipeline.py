import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.interpretability.gradcam import GradCAM

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

from preprocessing_pipeline import load_bonn_raw
from src.data.cwt import generate_cwt
from src.data.dataset import BonnDataset
from src.models.maven_net import MavenNet

from torch.utils.data import DataLoader


# ===============================
# Resize CAM
# ===============================
def resize_cam(cam, shape):
    cam = torch.as_tensor(cam).unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=shape, mode="bilinear", align_corners=False)
    cam = cam.squeeze().numpy()

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam


# ===============================
# Load Data
# ===============================
print("Loading Bonn dataset...")
data, labels, _ = load_bonn_raw("data/raw")
labels = (labels == 4).astype(np.int64)


# ===============================
# CWT
# ===============================
cwt_path = "data/processed/cwt_data.npy"

if not os.path.exists(cwt_path):
    print("Generating CWT...")
    cwt_data = np.array([generate_cwt(x) for x in data], dtype=np.float32)
    os.makedirs("data/processed", exist_ok=True)
    np.save(cwt_path, cwt_data)
else:
    cwt_data = np.load(cwt_path).astype(np.float32)


# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# Load Model
# ===============================
model = MavenNet().to(device)
model.load_state_dict(torch.load("checkpoints/ABCD_vs_E_fold_1.pth", map_location=device))
model.eval()


# ===============================
# Target Layer (validated)
# ===============================
if not hasattr(model, "conv_block_last"):
    raise AttributeError("Model has no 'conv_block_last'. Update target_layer.")

target_layer = model.conv_block_last


# ===============================
# Feature Extraction
# ===============================
def extract_features(model, X, y):

    loader = DataLoader(BonnDataset(X, y), batch_size=64)

    feats, lbls = [], []
    storage = []

    def hook_fn(module, input, output):
        storage.append(output.detach())

    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for x, label in loader:
            x = x.to(device)

            storage.clear()
            _ = model(x)

            f = storage[0]
            f = torch.mean(f, dim=[2, 3])

            feats.append(f.cpu().numpy())
            lbls.append(label.numpy())

    handle.remove()

    feats = np.concatenate(feats)
    lbls = np.concatenate(lbls)

    feats = StandardScaler().fit_transform(feats)

    return feats, lbls


# ===============================
# t-SNE
# ===============================
print("Running t-SNE...")

idx = np.random.choice(len(cwt_data), min(2000, len(cwt_data)), replace=False)
X_tsne = cwt_data[idx]
y_tsne = labels[idx]

features, _ = extract_features(model, X_tsne, y_tsne)

tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto", random_state=42)
coords = tsne.fit_transform(features)

plt.figure(figsize=(6, 5))
plt.scatter(coords[:, 0], coords[:, 1], c=y_tsne, cmap="coolwarm", s=10)
plt.title("t-SNE Feature Space")

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/tsne.png", dpi=200)
plt.close()


# ===============================
# Threshold (ROC)
# ===============================
loader = DataLoader(BonnDataset(cwt_data, labels), batch_size=64)

probs, y_true = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        p = torch.softmax(model(x), dim=1)[:, 1]
        probs.extend(p.cpu().numpy())
        y_true.extend(y.numpy())

probs = np.array(probs)
y_true = np.array(y_true)

fpr, tpr, thresholds = roc_curve(y_true, probs)
thr = thresholds[np.argmax(tpr - fpr)]

print("Threshold:", thr)
print("AUC:", auc(fpr, tpr))


# ===============================
# Grad-CAM
# ===============================
grad_cam = GradCAM(model, target_layer)

N = 5
seizure_idx = np.where(labels == 1)[0][:N]
normal_idx = np.where(labels == 0)[0][:N]

if len(seizure_idx) == 0 or len(normal_idx) == 0:
    raise RuntimeError("Missing samples for Grad-CAM visualization")


def run_cam(idx, title):
    sample = torch.from_numpy(cwt_data[idx:idx+1]).to(device)
    sample.requires_grad_(True)

    cam, _, prob = grad_cam.generate(sample)

    # threshold-aligned prediction
    pred = int(prob > thr)

    original = np.squeeze(cwt_data[idx])
    cam = resize_cam(cam, original.shape)

    plt.imshow(original, cmap='turbo', aspect='auto')
    plt.imshow(cam, cmap='jet', alpha=0.4)
    plt.title(f"{title} | Pred={pred} ({prob:.2f})")
    plt.axis("off")


plt.figure(figsize=(12, 6))

for i, idx in enumerate(seizure_idx):
    plt.subplot(2, N, i + 1)
    run_cam(idx, "Seizure")

for i, idx in enumerate(normal_idx):
    plt.subplot(2, N, N + i + 1)
    run_cam(idx, "Normal")

plt.savefig("outputs/gradcam.png", dpi=200)
plt.close()

# 🔴 IMPORTANT: remove hooks after use
grad_cam.remove_hooks()


# ===============================
# Confusion Matrix
# ===============================
preds = (probs > thr).astype(int)

cm = confusion_matrix(y_true, preds)

plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()

classes = ["Normal", "Seizure"]

plt.xticks([0, 1], classes)
plt.yticks([0, 1], classes)

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max()/2 else "black"
        )

plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("All outputs generated successfully.")