import torch
from torchinfo import summary
from model import MavenNet


if __name__ == "__main__":

    model = MavenNet()
    model.eval()

    print("\nMODEL SUMMARY:\n")

    summary(
        model,
        input_size=(1, 1, 224, 224),
        col_names=["output_size", "num_params"]
    )

    dummy_input = torch.randn(1, 1, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        "MavenNet.onnx",
        input_names=["Input"],
        output_names=["Output"],
        opset_version=18
    )

    print("\nONNX model exported as MavenNet.onnx")
    print("Upload to https://netron.app for clean architecture diagram.")