import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from preprocessing_pipeline import load_bonn_raw
from src.data.cwt import generate_cwt
from src.data.dataset import BonnDataset
from src.models.maven_net import MavenNet
from src.interpretability.gradcam import GradCAM

from torch.utils.data import DataLoader


# ===============================
# Resize CAM
# ===============================
def resize_cam(cam, shape):
    cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=shape, mode="bilinear", align_corners=False)
    return cam.squeeze().numpy()


# ===============================
# Load Data
# ===============================
print("Loading Bonn dataset...")
data, labels, _ = load_bonn_raw("data/raw")

# Convert to binary (IMPORTANT)
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
    cwt_data = np.load(cwt_path)


# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# Load Model
# ===============================
model = MavenNet().to(device)

# -------------------------------
# AUTO CHECKPOINT LOADING
# -------------------------------
ckpt_dir = "checkpoints"

if not os.path.exists(ckpt_dir):
    raise FileNotFoundError("checkpoints folder not found")

ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]

if len(ckpts) == 0:
    raise FileNotFoundError("No checkpoint files found")

# Sort for consistency (fold_1, fold_2, ...)
ckpts.sort()

# Pick first checkpoint (can improve later)
ckpt_path = os.path.join(ckpt_dir, ckpts[0])

print(f"Loading checkpoint: {ckpt_path}")

model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()


# ===============================
# Feature Extraction
# ===============================
def extract_features(model, X, y):

    loader = DataLoader(BonnDataset(X, y), batch_size=64)

    feats, lbls = [], []

    # 🔥 storage for hook
    feature_storage = []

    def hook_fn(module, input, output):
        feature_storage.append(output.detach())

    # ⚠️ attach hook to correct layer
    handle = model.attention.register_forward_hook(hook_fn)

    with torch.no_grad():
        for x, label in loader:
            x = x.to(device)

            feature_storage.clear()
            _ = model(x)

            f = feature_storage[0]
            f = torch.mean(f, dim=[2, 3])  # flatten

            feats.append(f.cpu().numpy())
            lbls.append(label.numpy())

    handle.remove()

    return np.concatenate(feats), np.concatenate(lbls)


RUN_TSNE_BEFORE = False
# ===============================
# t-SNE
# ===============================
print("Running t-SNE...")

# -------------------------------
# TSNE SAMPLING (safe)
# -------------------------------
MAX_SAMPLES = 2000

if len(cwt_data) > MAX_SAMPLES:
    idx = np.random.choice(len(cwt_data), MAX_SAMPLES, replace=False)
    cwt_data_tsne = cwt_data[idx]
    labels_tsne = labels[idx]
else:
    cwt_data_tsne = cwt_data
    labels_tsne = labels


# BEFORE
if RUN_TSNE_BEFORE:
    random_model = MavenNet().to(device)
    f_before, y_sample = extract_features(random_model, cwt_data_tsne, labels_tsne)

# AFTER
f_after, _ = extract_features(model, cwt_data_tsne, labels_tsne)

tsne = TSNE(n_components=2, random_state=42)

if RUN_TSNE_BEFORE:
    coords_before = tsne.fit_transform(f_before)

coords_after = tsne.fit_transform(f_after)

plt.figure(figsize=(10, 5))

if RUN_TSNE_BEFORE:
    plt.subplot(1, 2, 1)
    plt.title("t-SNE Before Training")
    plt.scatter(coords_before[:, 0], coords_before[:, 1], c=y_sample, cmap='coolwarm')

    plt.subplot(1, 2, 2)
    plt.title("t-SNE After Training")
    plt.scatter(coords_after[:, 0], coords_after[:, 1], c=y_sample, cmap='coolwarm')

else:
    plt.title("t-SNE After Training")
    plt.scatter(coords_after[:, 0], coords_after[:, 1], c=labels_tsne, cmap='coolwarm')

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/tsne_comparison.png", dpi=200)
plt.close()


# ===============================
# Grad-CAM
# ===============================
print("Generating Grad-CAM...")

grad_cam = GradCAM(model, model.attention)

seizure_idx = np.where(labels == 1)[0][0]
normal_idx = np.where(labels == 0)[0][0]


def run_cam(idx, title):
    sample = torch.tensor(cwt_data[idx:idx+1]).to(device)
    cam, pred, prob = grad_cam.generate(sample)

    original = cwt_data[idx][0]
    cam = resize_cam(cam, original.shape)

    plt.imshow(original, cmap='turbo', aspect='auto')
    plt.imshow(cam, cmap='jet', alpha=0.4)
    plt.title(f"{title} | Pred {pred} ({prob:.2f})")
    plt.axis("off")


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
run_cam(seizure_idx, "Seizure")

plt.subplot(1, 2, 2)
run_cam(normal_idx, "Normal")

plt.savefig("outputs/gradcam.png", dpi=200)
plt.close()


# ===============================
# Confusion Matrix
# ===============================
print("Computing confusion matrix...")

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

preds = (probs > 0.5).astype(int)

cm = confusion_matrix(y_true, preds)

plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()

classes = ["Normal", "Seizure"]

# Add ticks
plt.xticks([0, 1], classes)
plt.yticks([0, 1], classes)

# Add labels
plt.xlabel("Predicted")
plt.ylabel("Actual")

# 🔥 ADD NUMBERS INSIDE CELLS
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]),
        ha="center", va="center",
        color="white" if cm[i, j] > cm.max()/2 else "black")

plt.savefig("outputs/confusion_matrix.png")
plt.close()


print("All outputs generated successfully.")
required = [
    "outputs/tsne_comparison.png",
    "outputs/gradcam.png",
    "outputs/confusion_matrix.png"
]

for f in required:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing: {f}")