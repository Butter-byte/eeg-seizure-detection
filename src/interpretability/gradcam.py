import torch
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    # -----------------------------
    # Hooks
    # -----------------------------
    def _forward_hook(self, module, input, output):
        self.activations = output   # ✅ FIXED

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    # -----------------------------
    # Generate CAM
    # -----------------------------
    def generate(self, input_tensor, class_index=None):

        # ✅ RESET (important)
        self.gradients = None
        self.activations = None

        self.model.eval()

        output = self.model(input_tensor)

        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_class].item()

        if class_index is None:
            class_index = pred_class

        target = output[:, class_index]

        self.model.zero_grad(set_to_none=True)
        target.backward()

        # safety check
        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks failed")

        # -----------------------------
        # Grad-CAM
        # -----------------------------
        grads = self.gradients[0].detach().cpu().numpy()
        acts = self.activations[0].detach().cpu().numpy()

        grads_sq = grads ** 2
        grads_cube = grads_sq * grads

        eps = 1e-8
        denominator = 2 * grads_sq + np.sum(acts * grads_cube, axis=(1, 2), keepdims=True)
        denominator = np.where(denominator != 0, denominator, eps)

        alpha = grads_sq / denominator
        weights = np.sum(alpha * np.maximum(grads, 0), axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        # ReLU
        cam = np.maximum(cam, 0)

        # percentage normalization
        low = np.percentile(cam, 5)
        high = np.percentile(cam, 95)
        cam = np.clip((cam - low) / (high - low + 1e-8), 0, 1)

        # thresholding
        cam = np.where(cam > 0.4, cam, 0)

        return cam, pred_class, pred_prob

    # -----------------------------
    # Cleanup (IMPORTANT)
    # -----------------------------
    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()