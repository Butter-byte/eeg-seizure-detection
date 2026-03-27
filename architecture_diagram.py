import torch
from torchinfo import summary
from model import MavenNet


if __name__ == "__main__":

    # Force CPU usage
    device = torch.device("cpu")

    # Create model and move completely to CPU
    model = MavenNet()
    model = model.cpu()
    model.eval()

    print("\nMODEL SUMMARY:\n")

    summary(
        model,
        input_size=(1, 1, 178, 64),
        col_names=["output_size", "num_params"],
        device="cpu"
    )

    # Dummy input also on CPU
    dummy_input = torch.randn(1, 1, 178, 64)

    # Export to ONNX
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
        opset_version=18
    )

    print("\nONNX model exported as MavenNet.onnx")
    print("Upload it to https://netron.app for architecture visualization.")
