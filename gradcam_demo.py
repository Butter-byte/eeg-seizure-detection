import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from preprocessing_pipeline import load_bonn_csv
from model import MavenNet
from gradcam import GradCAM


# -------------------------------------------------
# Resize CAM to match original CWT size
# -------------------------------------------------
def resize_cam(cam, shape):
    cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam_resized = F.interpolate(
        cam_tensor,
        size=shape,
        mode="bilinear",
        align_corners=False
    )
    return cam_resized.squeeze().cpu().numpy()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    print("Loading dataset...")
    data, labels = load_bonn_csv("Epileptic Seizure Recognition.csv")

    print("Loading CWT representation...")
    cwt_data = np.load("cwt_data.npy")

    # Select one seizure and one normal sample
    seizure_idx = np.where(labels == 1)[0][0]
    normal_idx = np.where(labels == 0)[0][0]

    seizure_sample = torch.tensor(
        cwt_data[seizure_idx:seizure_idx+1],
        dtype=torch.float32
    ).to(device)

    normal_sample = torch.tensor(
        cwt_data[normal_idx:normal_idx+1],
        dtype=torch.float32
    ).to(device)

    # Load trained model
    model = MavenNet().to(device)
    print("Loading trained model...")
    model.load_state_dict(
        torch.load("best_model_fold_1.pth", map_location=device)
    )
    model.eval()

    # Hook last convolution layer BEFORE attention
    grad_cam = GradCAM(model, model.attention)

    # Generate GradCAM for seizure
    print("Generating seizure Grad-CAM...")
    cam_seizure, pred_s_class, pred_s_prob = grad_cam.generate(seizure_sample)

    # Generate GradCAM for normal
    print("Generating normal Grad-CAM...")
    cam_normal, pred_n_class, pred_n_prob = grad_cam.generate(normal_sample)

    # Normalize CAM
    cam_seizure = (cam_seizure - cam_seizure.min()) / (
        cam_seizure.max() - cam_seizure.min() + 1e-8
    )
    cam_normal = (cam_normal - cam_normal.min()) / (
        cam_normal.max() - cam_normal.min() + 1e-8
    )

    orig_seizure = cwt_data[seizure_idx][0]
    orig_normal = cwt_data[normal_idx][0]

    # Resize CAM to original shape
    cam_resized_seizure = resize_cam(cam_seizure, orig_seizure.shape)
    cam_resized_normal = resize_cam(cam_normal, orig_normal.shape)

    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------
    plt.figure(figsize=(12, 6))

    # Seizure original
    plt.subplot(2, 2, 1)
    plt.title("Seizure - CWT")
    plt.imshow(orig_seizure, aspect='auto', cmap='turbo')
    plt.xlabel("Time")
    plt.ylabel("Frequency Scale")
    plt.axis("off")

    # Seizure GradCAM
    plt.subplot(2, 2, 2)
    plt.title(f"Seizure - GradCAM\nPred: {pred_s_class} ({pred_s_prob:.3f})")
    plt.imshow(orig_seizure, aspect='auto', cmap='turbo')
    plt.imshow(cam_resized_seizure, cmap='jet', alpha=0.45)
    plt.xlabel("Time")
    plt.ylabel("Frequency Scale")
    plt.axis("off")

    # Normal original
    plt.subplot(2, 2, 3)
    plt.title("Normal - CWT")
    plt.imshow(orig_normal, aspect='auto', cmap='turbo')
    plt.xlabel("Time")
    plt.ylabel("Frequency Scale")
    plt.axis("off")

    # Normal GradCAM
    plt.subplot(2, 2, 4)
    plt.title(f"Normal - GradCAM\nPred: {pred_n_class} ({pred_n_prob:.3f})")
    plt.imshow(orig_normal, aspect='auto', cmap='turbo')
    plt.imshow(cam_resized_normal, cmap='jet', alpha=0.45)
    plt.xlabel("Time")
    plt.ylabel("Frequency Scale")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("Grad-CAM visualization complete.")