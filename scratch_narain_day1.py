import numpy as np
from src.model.autoencoder import build_lstm_autoencoder, make_windows

fake_data = np.random.rand(100, 4)
windows = make_windows(fake_data, window_size=10)

model = build_lstm_autoencoder(window_size=10, n_features=4, latent_dim=16)
model.fit(windows, windows, epochs=2, verbose=1)

model.save("scratch_model.keras")   # <-- this line is the whole point of your script