import sys
import os
import warnings
import logging

# -----------------------------
# Fix import path (VERY IMPORTANT)
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -----------------------------
# Silence logs
# -----------------------------
warnings.filterwarnings("ignore")
logging.getLogger("torch.onnx").setLevel(logging.ERROR)
os.environ["TORCH_LOGS"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"

# -----------------------------
# Imports
# -----------------------------
import torch
from torchinfo import summary
from src.models.maven_net import MavenNet


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    device = torch.device("cpu")  # keep CPU for export stability

    model = MavenNet().to(device)
    model.eval()

    print("\nMODEL SUMMARY:\n")

    # 🔥 IMPORTANT: match your training input (scales=32)
    summary(
        model,
        input_size=(1, 1, 178, 32),
        col_names=["output_size", "num_params"],
        device="cpu"
    )

    # Dummy input
    dummy_input = torch.randn(1, 1, 178, 32).to(device)

    # -----------------------------
    # ONNX EXPORT (CLEAN)
    # -----------------------------
    torch.onnx.export(
        model,
        dummy_input,
        "MavenNet.onnx",
        input_names=["Input"],
        output_names=["Output"],
        dynamic_axes={
            "Input": {0: "batch_size"},
            "Output": {0: "batch_size"}
        },
        opset_version=17,   # stable version
        verbose=False
    )

    print("\nONNX model exported as MavenNet.onnx")
    print("Open it in https://netron.app")