import pywt
import numpy as np

CENTRAL_FREQ = pywt.central_frequency('morl')


def generate_cwt(signal, scales=32, fs=173.61):
    """
    Convert a 1-D EEG segment to a CWT scalogram (1, T, scales).

    FIXES:
        - Guard against all-zero / degenerate signals before transform
        - Transposed output shape clarified: (1, T, scales) = (C, H, W)
          where H=time steps, W=frequency bins — consistent with dataset.py
        - Soft-clipping divisor tuned: tanh(x/2.5)*2.5 kept (good range)
        - Per-frequency std guard: avoid divide-by-zero on flat frequency rows
        - Added final finite-value guard (replaces NaN/Inf with 0)

    ACCURACY IMPROVEMENTS:
        - Frequency range kept at 4–35 Hz (clinically validated for seizure)
        - log1p amplification factor 2.0 kept (good empirical value)
        - Output dtype kept float32
    """

    # ── Input guard ───────────────────────────────────────────────────
    signal = np.asarray(signal, dtype=np.float64)
    if not np.isfinite(signal).all():
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(signal) < 1e-8:
        # Degenerate flat signal → return zero scalogram
        dummy = np.zeros((1, len(signal), scales), dtype=np.float32)
        return dummy

    # ── Frequency / scale grid ────────────────────────────────────────
    f_min = 4.0     # remove slow drift
    f_max = 35.0    # remove muscle / line noise

    frequencies  = np.geomspace(f_min, f_max, num=scales)
    scale_values = CENTRAL_FREQ * fs / frequencies

    # ── CWT ──────────────────────────────────────────────────────────
    coeffs, _ = pywt.cwt(signal, scale_values, 'morl')   # (scales, T)

    # ── Energy compression ───────────────────────────────────────────
    coeffs = np.abs(coeffs)
    coeffs = np.log1p(coeffs * 2.0)

    # ── Per-frequency (row) normalisation ────────────────────────────
    mean = coeffs.mean(axis=1, keepdims=True)
    std  = coeffs.std(axis=1,  keepdims=True)
    # FIX: guard zero-std rows individually
    std  = np.where(std < 1e-6, 1.0, std)
    coeffs = (coeffs - mean) / std

    # ── Soft clipping ────────────────────────────────────────────────
    coeffs = np.tanh(coeffs / 2.5) * 2.5

    # ── Final safety: replace any residual NaN/Inf ───────────────────
    coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=2.5, neginf=-2.5)

    # ── Shape: (1, T, scales)  →  (C=1, H=time, W=freq) ─────────────
    # coeffs is (scales, T) → transpose to (T, scales) → add channel
    coeffs = coeffs.T[np.newaxis, :, :].astype(np.float32)   # (1, T, scales)

    return coeffs
