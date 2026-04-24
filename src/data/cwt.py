import pywt
import numpy as np

CENTRAL_FREQ = pywt.central_frequency('morl')


def generate_cwt(signal, scales=32, fs=173.61):

    # -----------------------------
    # FREQUENCY RANGE (REFINED)
    # -----------------------------
    f_min = 2.0
    f_max = 60.0

    frequencies = np.geomspace(f_min, f_max, num=scales)
    scale_values = CENTRAL_FREQ * fs / frequencies

    coeffs, _ = pywt.cwt(signal, scale_values, 'morl')

    # -----------------------------
    # ENERGY COMPRESSION
    # -----------------------------
    coeffs = np.log1p(np.abs(coeffs))

    # -----------------------------
    # PER-FREQUENCY NORMALIZATION
    # -----------------------------
    mean = coeffs.mean(axis=1, keepdims=True)
    std = coeffs.std(axis=1, keepdims=True) + 1e-6
    coeffs = (coeffs - mean) / std

    # -----------------------------
    # SOFT CLIPPING
    # -----------------------------
    coeffs = np.tanh(coeffs / 3.0) * 3.0

    # -----------------------------
    # FORMAT FOR CNN
    # -----------------------------
    coeffs = coeffs.T[np.newaxis, :, :].astype(np.float32)

    return coeffs