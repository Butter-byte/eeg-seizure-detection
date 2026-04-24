import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.models.maven_net import MavenNet


def test_model():

    print("Running FULL model sanity check...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MavenNet().to(device)
    model.eval()

    x = torch.randn(1, 1, 178, 32).to(device)

    # -------------------------------
    # Forward check
    # -------------------------------
    with torch.no_grad():
        output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    assert output.shape == (1, 2), "❌ Output shape incorrect"

    # -------------------------------
    # Target layer existence
    # -------------------------------
    try:
        target_layer = model.conv_block_last
        print("\n✔ target_layer found: conv_block_last")
    except AttributeError:
        print("\n❌ conv_block_last not found")
        print("👉 You must change target_layer in your main script")
        print(model)
        return

    # -------------------------------
    # Feature map check
    # -------------------------------
    feature = None

    def forward_hook(module, input, output):
        nonlocal feature
        feature = output.detach()

    handle_fwd = target_layer.register_forward_hook(forward_hook)

    with torch.no_grad():
        _ = model(x)

    handle_fwd.remove()

    if feature is None:
        print("\n❌ Hook failed — no feature captured")
        return

    print("\nFeature map shape:", feature.shape)

    if len(feature.shape) != 4:
        print("❌ Not a valid conv feature map (needs [B,C,H,W])")
        return

    h, w = feature.shape[2], feature.shape[3]
    print(f"Feature map size: {h} x {w}")

    if h <= 1 or w <= 1:
        print("❌ Spatial size too small for Grad-CAM")
        return

    if h < 4 or w < 4:
        print("⚠️ Warning: very low spatial resolution → Grad-CAM may be weak")

    print("✔ Feature map valid for Grad-CAM")

    # -------------------------------
    # Gradient check at target layer
    # -------------------------------
    grad = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal grad
        grad = grad_output[0]

    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    x.requires_grad = True

    output = model(x)
    target = output[:, 1]

    model.zero_grad(set_to_none=True)
    target.backward()

    handle_bwd.remove()

    if grad is None:
        print("❌ No gradients at target layer (Grad-CAM will fail)")
        return

    print("Gradient shape:", grad.shape)

    if len(grad.shape) != 4:
        print("❌ Gradient shape invalid for Grad-CAM")
        return

    print("✔ Gradients exist at target layer")

    print("\n✅ FULL MODEL CHECK PASSED")


if __name__ == "__main__":
    test_model()