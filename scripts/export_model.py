import sys
import os
import warnings
import logging

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Silence logs
warnings.filterwarnings("ignore")
logging.getLogger("torch.onnx").setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
from torchinfo import summary
from src.models.maven_net import MavenNet


if __name__ == "__main__":

    device = torch.device("cpu")

    model = MavenNet().to(device)
    model.eval()

    print("\nMODEL SUMMARY:\n")

    summary(
        model,
        input_size=(1, 1, 178, 32),
        col_names=["output_size", "num_params"],
        device="cpu"
    )

    dummy_input = torch.randn(1, 1, 178, 32).to(device)

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
        opset_version=17,
        verbose=False
    )

    print("\nONNX model exported as MavenNet.onnx")