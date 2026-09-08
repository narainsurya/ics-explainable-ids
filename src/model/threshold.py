"""
Turns raw reconstruction error into a NORMAL/ANOMALY decision.

WHAT: Given reconstruction errors from normal validation data, picks a
      cutoff value. Any window with error above the cutoff is flagged.

WHY: The autoencoder alone only outputs a number (how bad the reconstruction
     was). Something has to turn that into a yes/no decision -- that's this
     file's job. Narain's SHAP explainer only runs on windows THIS file
     flags as anomalies.
"""

import numpy as np


def compute_reconstruction_error(model, windows: np.ndarray) -> np.ndarray:
    """
    Returns one error score per window: mean squared error between the
    original window and the model's reconstruction of it.
    """
    reconstructions = model.predict(windows, verbose=0)
    mse = np.mean(np.square(windows - reconstructions), axis=(1, 2))
    return mse


def pick_threshold(normal_errors: np.ndarray, k: float = 3.0) -> float:
    """
    Picks threshold = mean(normal_errors) + k * std(normal_errors).

    Tuning k:
      - k too LOW  -> almost everything gets flagged (too many false alarms)
      - k too HIGH -> almost nothing gets flagged (misses real attacks)
      Start with k=3.0 and adjust after looking at evaluate.py's precision/recall.
    """
    mean = np.mean(normal_errors)
    std = np.std(normal_errors)
    threshold = mean + k * std
    print(f"Normal error mean={mean:.6f}, std={std:.6f}, threshold={threshold:.6f}")
    return threshold


def classify(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Returns an array of 0/1 labels: 1 = ANOMALY, 0 = NORMAL."""
    return (errors > threshold).astype(int)