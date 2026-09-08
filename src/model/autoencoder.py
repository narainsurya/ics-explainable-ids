"""
LSTM Autoencoder for ICS sensor anomaly detection.

WHAT: Defines the model architecture (build_lstm_autoencoder) and a helper
      to convert raw sensor rows into overlapping time windows (make_windows),
      which is the input shape LSTMs expect: (samples, timesteps, features).

WHY: The autoencoder is trained ONLY on normal data. It learns to compress
     and then reconstruct normal pipeline behavior. When it later sees an
     attack, it reconstructs it badly -> high "reconstruction error" ->
     that error is what threshold.py turns into an ANOMALY flag.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def make_windows(data: np.ndarray, window_size: int = 30, step: int = 1) -> np.ndarray:
    """
    Slice a 2D array of sensor readings (rows=timesteps, cols=features)
    into overlapping windows of shape (window_size, features).

    Example: 1000 rows, 3 features, window_size=30 -> ~970 windows of shape (30, 3)
    """
    n_timesteps, n_features = data.shape
    windows = []
    for start in range(0, n_timesteps - window_size + 1, step):
        windows.append(data[start:start + window_size])
    return np.array(windows)


def build_lstm_autoencoder(window_size: int, n_features: int, latent_dim: int = 16) -> keras.Model:
    """
    Encoder: LSTM layers compress the window down to a small latent vector.
    Decoder: LSTM layers try to rebuild the original window from that vector.
    Trained to minimize reconstruction error (MSE) on NORMAL data only.
    """
    inputs = keras.Input(shape=(window_size, n_features))

    # --- Encoder ---
    x = layers.LSTM(64, activation="tanh", return_sequences=True)(inputs)
    x = layers.LSTM(latent_dim, activation="tanh", return_sequences=False)(x)

    # --- Bottleneck, repeated across timesteps so decoder LSTM gets a sequence ---
    x = layers.RepeatVector(window_size)(x)

    # --- Decoder ---
    x = layers.LSTM(latent_dim, activation="tanh", return_sequences=True)(x)
    x = layers.LSTM(64, activation="tanh", return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


if __name__ == "__main__":
    # Day 1 self-test: fake random data, just proving the pipeline runs end to end.
    fake_data = np.random.rand(500, 3)          # 500 fake timesteps, 3 sensors
    windows = make_windows(fake_data, window_size=30)
    print("Window shape:", windows.shape)         # (471, 30, 3)

    model = build_lstm_autoencoder(window_size=30, n_features=3)
    model.summary()

    history = model.fit(windows, windows, epochs=3, batch_size=16, verbose=1)
    print("Final loss:", history.history["loss"][-1])