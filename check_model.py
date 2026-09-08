import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import shap

model = load_model("scratch_model.keras")

inputs = model.input
reconstruction = model.output

# No keepdims -> shape becomes (batch,), true rank 1
error = tf.keras.layers.Lambda(
    lambda t: tf.reduce_mean(tf.square(t[0] - t[1]), axis=[1, 2])
)([inputs, reconstruction])

# Reshape to (batch, 1) - a clean "vector" output SHAP expects
error = tf.keras.layers.Reshape((1,))(error)

error_model = tf.keras.Model(inputs, error)

background = np.random.rand(10, 10, 4).astype("float32")
explainer = shap.GradientExplainer(error_model, background)

test_window = np.random.rand(1, 10, 4).astype("float32")
shap_values = explainer.shap_values(test_window)

print("Success!")
print(type(shap_values), np.array(shap_values).shape)