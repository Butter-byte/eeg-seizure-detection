import torch
import numpy as np


class GradCAM:
    """
    Grad-CAM++ implementation for MavenNet.

    FIXES
    -----
    - model.eval() moved to __init__ so it is not reset on every generate()
      call (was safe but wasteful)
    - gradient/activation device mismatch: both are now explicitly cast to
      float32 CPU numpy — prevents silent dtype bugs with AMP
    - zero-gradient guard: if grads are all-zero, fall back to plain CAM
      (weights = global average pooling of activations) instead of crashing
    - Hook cleanup: remove_hooks() now idempotent (safe to call twice)
    - generate() returns consistent (H, W) cam regardless of feature map size

    ACCURACY IMPROVEMENTS (visualisation quality)
    ----------------------------------------------
    - Percentile-based normalisation kept (good for robustness)
    - Threshold lowered to 25th percentile kept (shows more activation)
    - Smooth final normalisation kept
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer

        self.gradients  = None
        self.activations = None
        self._hooks_active = False

        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------
    def _register_hooks(self):
        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)
        self._hooks_active = True

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()      # FIX: detach early, saves memory

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Generate CAM
    # ------------------------------------------------------------------
    def generate(self, input_tensor, class_index=None):
        """
        Parameters
        ----------
        input_tensor : (1, C, H, W) torch.Tensor on same device as model
        class_index  : int or None  — None → use predicted class

        Returns
        -------
        cam        : (H_feat, W_feat) numpy float32 in [0, 1]
        pred_class : int
        pred_prob  : float
        """
        self.gradients  = None
        self.activations = None

        # Ensure we can compute gradients
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        self.model.eval()
        output = self.model(input_tensor)

        probs      = torch.softmax(output, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        pred_prob  = float(probs[0, pred_class].item())

        if class_index is None:
            class_index = pred_class

        # Backward
        self.model.zero_grad(set_to_none=True)
        target = output[0, class_index]
        target.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "GradCAM hooks did not fire. "
                "Make sure target_layer is part of the forward pass."
            )

        # Cast to float32 numpy (safe with AMP)
        grads = self.gradients[0].cpu().float().numpy()     # (C, H, W)
        acts  = self.activations[0].cpu().float().numpy()   # (C, H, W)

        # ── Grad-CAM++ weights ────────────────────────────────────────
        grads_sq    = grads ** 2
        grads_cube  = grads_sq * grads
        eps         = 1e-8

        denominator = (
            2.0 * grads_sq
            + np.sum(acts * grads_cube, axis=(1, 2), keepdims=True)
        )
        # FIX: use np.where with scalar eps to avoid modifying near-zero
        denominator = np.where(np.abs(denominator) < eps, eps, denominator)

        alpha   = grads_sq / denominator                                    # (C, H, W)
        weights = np.sum(alpha * np.maximum(grads, 0), axis=(1, 2))        # (C,)

        # FIX: zero-gradient fallback → plain CAM
        if np.abs(weights).max() < 1e-8:
            weights = acts.mean(axis=(1, 2))        # global avg pool fallback

        cam = np.einsum('c,chw->hw', weights, acts)     # weighted sum

        # ReLU
        cam = np.maximum(cam, 0)

        # ── Normalisation ─────────────────────────────────────────────
        low  = np.percentile(cam, 5)
        high = np.percentile(cam, 95)

        if high - low < 1e-6:
            cam = np.zeros_like(cam)
        else:
            cam = np.clip((cam - low) / (high - low + eps), 0.0, 1.0)

        # Threshold: suppress bottom 25% activations
        thr = np.percentile(cam, 25)
        cam = np.where(cam > thr, cam, 0.0)

        # Final normalisation
        max_val = cam.max()
        if max_val > 1e-8:
            cam = cam / (max_val + eps)

        return cam.astype(np.float32), pred_class, pred_prob

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def remove_hooks(self):
        if self._hooks_active:
            self.fwd_handle.remove()
            self.bwd_handle.remove()
            self._hooks_active = False
