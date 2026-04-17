import pywt
import numpy as np

def generate_cwt(signal, scales=32):  # ↓ reduced from 64 → faster
    coeffs, _ = pywt.cwt(signal, np.arange(1, scales + 1), 'morl')

    coeffs = np.abs(coeffs)

    mean = coeffs.mean(axis=1, keepdims=True)
    std = coeffs.std(axis=1, keepdims=True)
    std[std == 0] = 1

    coeffs = (coeffs - mean) / std

    return coeffs.T[np.newaxis, :, :].astype(np.float32)