import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from src.models.maven_net import MavenNet


def test_model():

    print("Running model sanity check...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MavenNet().to(device)
    model.eval()

    # Dummy input (match training shape)
    x = torch.randn(1, 1, 178, 32).to(device)

    with torch.no_grad():
        output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    # Expected output = (batch_size, num_classes)
    assert output.shape == (1, 2), "❌ Output shape incorrect"

    print("\n✅ Model forward pass OK")


if __name__ == "__main__":
    test_model()