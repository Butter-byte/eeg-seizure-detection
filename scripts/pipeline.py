import os
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
data, labels = load_bonn_raw(".")

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


model.load_state_dict(torch.load("checkpoints/best_model_ABCD_vs_E_fold1.pth", map_location=device))

model.eval()


# ===============================
# Feature Extraction
# ===============================
def extract_features(model, X, y):
    loader = DataLoader(BonnDataset(X, y), batch_size=64)
    feats, lbls = [], []

    with torch.no_grad():
        for x, label in loader:
            x = x.to(device)
            _, f = model(x, return_features=True)
            feats.append(f.cpu().numpy())
            lbls.append(label.numpy())

    return np.concatenate(feats), np.concatenate(lbls)


# ===============================
# t-SNE
# ===============================
print("Running t-SNE...")

# BEFORE
random_model = MavenNet().to(device)
f_before, y_sample = extract_features(random_model, cwt_data, labels)

# AFTER
f_after, _ = extract_features(model, cwt_data, labels)

tsne = TSNE(n_components=2, random_state=42)

coords_before = tsne.fit_transform(f_before)
coords_after = tsne.fit_transform(f_after)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("t-SNE Before Training")
plt.scatter(coords_before[:, 0], coords_before[:, 1], c=y_sample, cmap='coolwarm')

plt.subplot(1, 2, 2)
plt.title("t-SNE After Training")
plt.scatter(coords_after[:, 0], coords_after[:, 1], c=y_sample, cmap='coolwarm')

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

plt.savefig("outputs/confusion_matrix.png")
plt.close()


print("All outputs generated successfully.")