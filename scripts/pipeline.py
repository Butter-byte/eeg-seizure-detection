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

RUN_TSNE = False
RUN_ROC = True
RUN_GRADCAM = True
RUN_CM = False

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
def get_last_conv_layer(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in model")

target_layer = get_last_conv_layer(model)


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
# ===============================\
if RUN_TSNE:
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
if RUN_ROC:
    print("Calculating ROC curve...")
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
# Grad-CAM (3 Seizure Samples)
# ===============================
if RUN_GRADCAM:

    grad_cam = GradCAM(model, target_layer)

    # Pick 3 seizure samples
    seizure_idx = np.where(labels == 1)[0][:3]

    plt.figure(figsize=(8, 10))

    for i, idx in enumerate(seizure_idx):

        sample = torch.from_numpy(cwt_data[idx:idx+1]).to(device)
        sample.requires_grad_(True)

        # Prediction
        with torch.no_grad():
            out = model(sample)
            prob = torch.softmax(out, dim=1)[0, 1].item()

        pred = int(prob > thr)

        # Grad-CAM (force seizure class)
        cams = []

        for _ in range(5):   # try 5 first
            noise = torch.randn_like(sample) * 0.01
            cam_i, _, _ = grad_cam.generate(sample + noise, class_index=1)
            cams.append(cam_i)

        cam = np.mean(cams, axis=0)

        original = np.squeeze(cwt_data[idx])
        cam = resize_cam(cam, original.shape)

        # -----------------------
        # Original
        # -----------------------
        plt.subplot(3, 2, i*2 + 1)
        im1 = plt.imshow(original, cmap='turbo', aspect='auto', interpolation='bilinear')
        plt.title(f"Seizure {i+1} - Original")
        plt.axis("off")
        plt.colorbar(im1, fraction=0.046, pad=0.02)

        # -----------------------
        # Grad-CAM
        # -----------------------
        plt.subplot(3, 2, i*2 + 2)
        plt.imshow(original, cmap='turbo', aspect='auto', interpolation='bilinear')
        im2 = plt.imshow(cam, cmap='jet', alpha=0.4, interpolation='bilinear')
        plt.title(f"Grad-CAM Overlay \nPred={pred} ({prob:.2f})")
        plt.axis("off")
        plt.colorbar(im2, fraction=0.046, pad=0.02)

    plt.subplots_adjust(top=0.92)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/gradcam_seizure_3samples.png", dpi=200)
    plt.close()

    grad_cam.remove_hooks()


# ===============================
# Confusion Matrix
# ===============================
if RUN_CM:
    print("Generating confusion matrix...")
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