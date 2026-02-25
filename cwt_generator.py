import pywt
import numpy as np


def generate_cwt(signal, scales=64):

    # Compute CWT
    coefficients, _ = pywt.cwt(
        signal,
        np.arange(1, scales + 1),
        'morl'
    )

    # Magnitude
    coefficients = np.abs(coefficients)

    # Normalize per-scale (better stability)
    mean = coefficients.mean(axis=1, keepdims=True)
    std = coefficients.std(axis=1, keepdims=True)

    std[std == 0] = 1

    coefficients = (coefficients - mean) / std

    # Shape → (1, time, scales)
    coefficients = coefficients.T[np.newaxis, :, :]

    return coefficients.astype(np.float32)