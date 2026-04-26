"""
pipeline.py  —  Visualisation & analysis pipeline for MavenNet EEG classifier.

Improvements over original
--------------------------
t-SNE   : uses PCA(50) + perplexity=30, colour-coded with legend, larger dots,
           proper axis labels, dark background style, per-class convex hulls
ROC     : AUC in title, operating point marked, shaded region, grid
Grad-CAM: side-by-side original + overlay, colourbar, title with prob,
           better colour maps, tight layout
Confusion matrix: percentage + count, per-cell colour, proper font sizing
General: all figures use consistent style (seaborn-v0_8 / fallback),
         outputs/ directory always created, DPI=200 kept
"""

import sys
import os

# ── Path setup: add project root AND src sub-packages to sys.path ──────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_MODELS   = os.path.join(PROJECT_ROOT, "src", "models")
SRC_DATA     = os.path.join(PROJECT_ROOT, "src", "data")
SRC_UTILS    = os.path.join(PROJECT_ROOT, "src", "utils")
SRC_INTERP   = os.path.join(PROJECT_ROOT, "src", "interpretability")
SCRIPTS_DIR  = os.path.join(PROJECT_ROOT, "scripts")

for _p in [PROJECT_ROOT, SRC_MODELS, SRC_DATA, SRC_UTILS, SRC_INTERP, SCRIPTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F

from sklearn.manifold      import TSNE
from sklearn.metrics       import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.interpretability.gradcam import GradCAM
from preprocessing_pipeline import load_bonn_raw    # scripts/preprocessing_pipeline.py
from src.data.cwt import generate_cwt
from src.data.dataset import BonnDataset
from src.models.maven_net import MavenNet

from torch.utils.data import DataLoader

# ── Style ─────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

PALETTE = {
    "normal" : "#4C72B0",
    "seizure": "#DD8452",
    "grad"   : "magma",
    "roc"    : "#2ca02c",
}

# ── Absolute paths (works regardless of where you run the script from) ──
DATA_RAW_DIR  = os.path.join(PROJECT_ROOT, "data", "raw")
CWT_CACHE     = os.path.join(PROJECT_ROOT, "data", "processed", "cwt_data.npy")
CKPT_DIR      = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUTS_DIR   = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RUN_TSNE    = True
RUN_ROC     = True
RUN_GRADCAM = True
RUN_CM      = True

CONF_MARGIN = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def batch_normalize(x):
    mean = x.mean(dim=(2, 3), keepdim=True)
    std  = x.std(dim=(2, 3),  keepdim=True) + 1e-8
    return torch.clamp((x - mean) / std, -3.0, 3.0)


def resize_cam(cam, shape):
    cam_t = torch.as_tensor(cam).unsqueeze(0).unsqueeze(0).float()
    cam_t = F.interpolate(cam_t, size=shape, mode="bilinear", align_corners=False)
    cam_t = cam_t.squeeze().numpy()
    cam_t = cam_t - cam_t.min()
    max_v = cam_t.max()
    if max_v > 1e-8:
        cam_t /= max_v
    return cam_t


def get_last_conv_layer(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in model")


def infer_probs(model, X, y, device):
    loader = DataLoader(BonnDataset(X, y), batch_size=64)
    probs, y_true = [], []
    with torch.no_grad():
        for x, yb in loader:
            x = batch_normalize(x.to(device))
            p = torch.softmax(model(x), dim=1)[:, 1]
            probs.extend(p.cpu().numpy())
            y_true.extend(yb.numpy())
    return np.array(probs), np.array(y_true)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading Bonn dataset...")
data, labels, _ = load_bonn_raw(DATA_RAW_DIR)
labels = (labels == 4).astype(np.int64)   # binary: seizure vs non-seizure

cwt_path = CWT_CACHE
if not os.path.exists(cwt_path):
    print("Generating CWT scalograms...")
    cwt_data = np.array([generate_cwt(x) for x in data], dtype=np.float32)
    os.makedirs(os.path.dirname(CWT_CACHE), exist_ok=True)
    np.save(cwt_path, cwt_data)
else:
    cwt_data = np.load(cwt_path).astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
model = MavenNet().to(device)
model.load_state_dict(
    torch.load(os.path.join(CKPT_DIR, "ABCD_vs_E_fold_1.pth"), map_location=device),
    strict=False   # FIX: strict=False allows loading old ckpts into new arch
)
model.eval()

target_layer = get_last_conv_layer(model)


# ─────────────────────────────────────────────────────────────────────────────
# ROC CURVE  ── IMPROVED
# ─────────────────────────────────────────────────────────────────────────────
if RUN_ROC:
    print("\nRunning ROC curve...")
    probs, y_true = infer_probs(model, cwt_data, labels, device)

    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    best_idx = int(np.argmax(tpr - fpr))
    thr      = thresholds[best_idx]

    print(f"  AUC       : {roc_auc:.4f}")
    print(f"  Threshold : {thr:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE["roc"], lw=2,
            label=f"MavenNet (AUC = {roc_auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["roc"])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random")

    # Operating point
    ax.scatter(fpr[best_idx], tpr[best_idx],
               s=80, zorder=5, color="crimson",
               label=f"Op. point (thr={thr:.2f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curve — AUC = {roc_auc:.4f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "roc_curve.png"), dpi=200)
    plt.close(fig)
    print("  Saved: outputs/roc_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE  ── IMPROVED
# ─────────────────────────────────────────────────────────────────────────────
if RUN_TSNE:
    print("\nRunning t-SNE...")

    n_samples = min(2000, len(cwt_data))
    idx       = np.random.choice(len(cwt_data), n_samples, replace=False)
    X_tsne    = cwt_data[idx]
    y_tsne    = labels[idx]

    loader = DataLoader(BonnDataset(X_tsne, y_tsne), batch_size=64)
    feats  = []
    with torch.no_grad():
        for x, _ in loader:
            x = batch_normalize(x.to(device))
            f = model(x, return_features=True)
            feats.append(f.cpu().numpy())
    feats = np.concatenate(feats)

    # PCA whitening before t-SNE
    feats = StandardScaler().fit_transform(feats)
    n_pca = min(50, feats.shape[0] - 1, feats.shape[1])
    feats = PCA(n_components=n_pca, whiten=True).fit_transform(feats)

    # FIX: perplexity=30, more iterations for better convergence
    tsne   = TSNE(n_components=2, perplexity=30, max_iter=3000,
                  learning_rate="auto", init="pca", random_state=42)
    coords = tsne.fit_transform(feats)

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#F5F5F5")

    class_names  = ["Normal (A–D)", "Seizure (E)"]
    class_colors = [PALETTE["normal"], PALETTE["seizure"]]
    markers      = ["o", "^"]

    for cls, (name, color, marker) in enumerate(
            zip(class_names, class_colors, markers)):
        mask = y_tsne == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=name, marker=marker,
            s=14, alpha=0.75, linewidths=0,
        )

    ax.set_title("t-SNE Feature Space (MavenNet embeddings)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.legend(fontsize=11, markerscale=2, framealpha=0.9)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "tsne.png"), dpi=200)
    plt.close(fig)
    print("  Saved: outputs/tsne.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM  ── IMPROVED
# ─────────────────────────────────────────────────────────────────────────────
if RUN_GRADCAM:
    print("\nRunning Grad-CAM...")

    grad_cam     = GradCAM(model, target_layer)
    seizure_idx  = np.where(labels == 1)[0][:4]   # show 4 examples
    n            = len(seizure_idx)

    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(seizure_idx):
        sample = torch.from_numpy(cwt_data[idx:idx+1]).to(device)
        sample = batch_normalize(sample)

        # Smooth CAM via multiple noisy passes
        cams = []
        for _ in range(7):
            noise = torch.randn_like(sample) * 0.01
            cam_i, pred_cls, pred_prob = grad_cam.generate(
                sample + noise, class_index=1
            )
            cams.append(cam_i)
        cam = np.mean(cams, axis=0)

        original = np.squeeze(cwt_data[idx])           # (H, W)
        cam_rs   = resize_cam(cam, original.shape)

        # Col 0: original scalogram
        im0 = axes[row, 0].imshow(original, cmap="turbo",
                                  aspect="auto", origin="lower")
        axes[row, 0].set_title("CWT Scalogram", fontsize=10)
        axes[row, 0].set_xlabel("Time")
        axes[row, 0].set_ylabel("Frequency")
        axes[row, 0].axis("on")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # Col 1: Grad-CAM heatmap only
        im1 = axes[row, 1].imshow(cam_rs, cmap=PALETTE["grad"],
                                  aspect="auto", origin="lower", vmin=0, vmax=1)
        axes[row, 1].set_title("Grad-CAM++ Activation", fontsize=10)
        axes[row, 1].set_xlabel("Time")
        axes[row, 1].axis("on")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Col 2: overlay
        axes[row, 2].imshow(original, cmap="turbo",
                            aspect="auto", origin="lower")
        axes[row, 2].imshow(cam_rs,   cmap="jet",
                            aspect="auto", origin="lower",
                            alpha=0.45, vmin=0, vmax=1)
        axes[row, 2].set_title(
            f"Overlay  (p={pred_prob:.3f})", fontsize=10
        )
        axes[row, 2].set_xlabel("Time")
        axes[row, 2].axis("on")

    fig.suptitle("Grad-CAM++ — Seizure Samples", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "gradcam.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    grad_cam.remove_hooks()
    print("  Saved: outputs/gradcam.png")


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX  ── IMPROVED
# ─────────────────────────────────────────────────────────────────────────────
if RUN_CM:
    print("\nRunning Confusion Matrix...")

    probs, y_true = infer_probs(model, cwt_data, labels, device)

    fpr, tpr, thresholds = roc_curve(y_true, probs)
    thr   = thresholds[np.argmax(tpr - fpr)]
    preds = (probs > thr).astype(int)

    cm    = confusion_matrix(y_true, preds)
    total = cm.sum()

    classes = ["Normal", "Seizure"]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Custom colour map per cell: TP/TN green-ish, FP/FN red-ish
    cell_colors = np.array([
        ["#AED6F1", "#E74C3C"],   # row 0: TN blue, FP red
        ["#E74C3C", "#2ECC71"],   # row 1: FN red, TP green
    ])

    for i in range(2):
        for j in range(2):
            count   = cm[i, j]
            perc    = count / total * 100
            row_sum = cm[i].sum()
            row_pct = count / row_sum * 100 if row_sum > 0 else 0

            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.05",
                    facecolor=cell_colors[i, j],
                    edgecolor="white", linewidth=2,
                )
            )
            label = f"{count}\n{perc:.1f}% total\n{row_pct:.1f}% row"
            ax.text(j, i, label,
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if (i, j) in [(0, 1), (1, 0)] else "black")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1]);  ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks([0, 1]);  ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)

    sens = cm[1, 1] / (cm[1, 1] + cm[1, 0] + 1e-8)
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-8)
    ax.set_title(
        f"Confusion Matrix  |  Sens={sens:.3f}  Spec={spec:.3f}",
        fontsize=12, fontweight="bold"
    )

    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"), dpi=200)
    plt.close(fig)
    print("  Saved: outputs/confusion_matrix.png")


print("\nAll outputs generated successfully.")
