"""
Trains the real LSTM Autoencoder on Ashish's real, scaled normal data.

WHAT: Loads data/processed/train_normal.csv, builds windows, trains the
      autoencoder, picks a threshold, and saves both the model and a
      config.json with everything needed to use it later.

WHY: This produces the ONE trained model file that Narain's SHAP explainer
     and Pranesh's dashboard both load.

Run: python -m src.model.train
"""

import json
import os
import numpy as np
import pandas as pd

from src.model.autoencoder import build_lstm_autoencoder, make_windows
from src.model.threshold import compute_reconstruction_error, pick_threshold

# --- Edit these if Ashish's real column names/paths differ from the plan ---
FEATURE_COLUMNS = ["pressure", "flow_rate", "temperature"]
TRAIN_DATA_PATH = "data/processed/train_scaled.csv"
WINDOW_SIZE = 30
EPOCHS = 20
BATCH_SIZE = 32
K_THRESHOLD = 3.0          # <- the "k value" you adjust if flagging looks off
MODEL_DIR = "models/saved"


def load_normal_data() -> np.ndarray:
    df = pd.read_csv(TRAIN_DATA_PATH)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns from Ashish's data: {missing}")
    return df[FEATURE_COLUMNS].to_numpy(dtype="float32")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading normal training data...")
    data = load_normal_data()
    windows = make_windows(data, window_size=WINDOW_SIZE)
    print(f"Built {windows.shape[0]} windows of shape {windows.shape[1:]}")

    # Hold out 10% of normal windows, untouched by training, to pick the
    # threshold on data the model hasn't memorized.
    split = int(0.9 * len(windows))
    train_windows, val_windows = windows[:split], windows[split:]

    print("Building model...")
    model = build_lstm_autoencoder(window_size=WINDOW_SIZE, n_features=len(FEATURE_COLUMNS))
    model.summary()

    print("Training...")
    history = model.fit(
        train_windows, train_windows,
        validation_data=(val_windows, val_windows),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    print("Final train loss:", history.history["loss"][-1])

    print("Picking threshold from validation (normal) reconstruction error...")
    val_errors = compute_reconstruction_error(model, val_windows)
    threshold = pick_threshold(val_errors, k=K_THRESHOLD)

    print("Saving model and config...")
    model.save(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))
    config = {
        "window_size": WINDOW_SIZE,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": float(threshold),
        "k_threshold": K_THRESHOLD,
    }
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Done. Model + config saved to", MODEL_DIR)
    print("Tell Narain and Pranesh the model is ready to load.")


if __name__ == "__main__":
    main()