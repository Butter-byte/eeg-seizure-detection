import torch
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_index=None):

        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_class].item()

        if class_index is None:
            class_index = pred_class

        # Backward pass
        target = output[0, class_index]

        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Extract gradients and activations
        gradients = self.gradients[0].detach().cpu().numpy()
        activations = self.activations[0].detach().cpu().numpy()

        # Compute channel-wise weights
        weights = np.mean(gradients, axis=(1, 2))

        # Weighted sum
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = np.maximum(cam, 0)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, pred_class, pred_prob