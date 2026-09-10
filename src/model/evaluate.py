"""
Scores the frozen model against the FULL labeled test set.

WHAT: Loads data/processed/test_labeled.csv (both normal and attack rows,
      with a ground-truth `label` column), runs it through the trained
      model, and reports precision, recall, F1, and ROC-AUC.

WHY: This is the number you say out loud when a visitor asks "how accurate
     is it?" Also your last chance to catch a badly-tuned threshold before
     freezing the model.

Run: python -m src.model.evaluate
"""

import json
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

from src.model.autoencoder import make_windows
from src.model.threshold import compute_reconstruction_error, classify

TEST_DATA_PATH = "data/processed/test_scaled.csv"
MODEL_PATH = "models/saved/lstm_autoencoder.h5"
CONFIG_PATH = "models/saved/config.json"


def window_labels(labels: np.ndarray, window_size: int) -> np.ndarray:
    """
    A window is labeled ANOMALY if ANY reading inside it was part of an
    attack. Must use the same window_size/stride as make_windows.
    """
    n = len(labels) - window_size + 1
    return np.array([1 if labels[i:i + window_size].max() > 0 else 0 for i in range(n)])


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    model = keras.models.load_model(MODEL_PATH)
    df = pd.read_csv(TEST_DATA_PATH)

    data = df[config["feature_columns"]].to_numpy(dtype="float32")
    labels = df["label"].to_numpy()

    windows = make_windows(data, window_size=config["window_size"])
    true_labels = window_labels(labels, config["window_size"])

    errors = compute_reconstruction_error(model, windows)
    predictions = classify(errors, config["threshold"])

    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    try:
        auc = roc_auc_score(true_labels, errors)
    except ValueError:
        auc = float("nan")  # only one class present in this test slice

    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(true_labels, predictions))
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")

    print("\nIf recall is low (missing attacks) -> lower k in train.py, retrain.")
    print("If precision is low (false alarms) -> raise k in train.py, retrain.")


if __name__ == "__main__":
    main()