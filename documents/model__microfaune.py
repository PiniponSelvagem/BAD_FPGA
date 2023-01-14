from tensorflow import keras
from keras import layers
from tensorflow import math
import numpy as np

n_filter = 64

spec = layers.Input(shape=[None, 40, 1], dtype=np.float32)
x = layers.Conv2D(n_filter, (3, 3), padding="same",
                    activation=None)(spec)
x = layers.BatchNormalization(momentum=0.95)(x)
x = layers.ReLU()(x)
x = layers.Conv2D(n_filter, (3, 3), padding="same",
                    activation=None)(x)
x = layers.BatchNormalization(momentum=0.95)(x)
x = layers.ReLU()(x)
x = layers.MaxPool2D((1, 2))(x)

x = layers.Conv2D(n_filter, (3, 3), padding="same",
                    activation=None)(x)
x = layers.BatchNormalization(momentum=0.95)(x)
x = layers.ReLU()(x)
x = layers.Conv2D(n_filter, (3, 3), padding="same",
                    activation=None)(x)
x = layers.BatchNormalization(momentum=0.95)(x)
x = layers.ReLU()(x)
x = layers.MaxPool2D((1, 2))(x)

x = layers.Conv2D(n_filter, (3, 3), padding="same",
                    activation=None)(x)
x = layers.BatchNormalization(momentum=0.95)(x)
x = layers.ReLU()(x)
x = layers.Conv2D(n_filter, (3, 3), padding="same",
                    activation=None)(x)
x = layers.BatchNormalization(momentum=0.95)(x)
x = layers.ReLU()(x)
x = layers.MaxPool2D((1, 2))(x)

x = math.reduce_max(x, axis=-2)

x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)

x = layers.TimeDistributed(layers.Dense(64, activation="sigmoid"))(x)
local_pred = layers.TimeDistributed(
    layers.Dense(1, activation="sigmoid"))(x)

pred = math.reduce_max(local_pred, axis=-2)
model = keras.Model(inputs=spec, outputs=[pred, local_pred])



model.summary()