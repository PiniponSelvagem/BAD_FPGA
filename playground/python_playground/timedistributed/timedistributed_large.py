import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#########################################################

CUDA_VISIBLE_DEVICES=""     # force use CPU

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

spec = layers.Input(shape=[None, 128], dtype=np.float32)
x = layers.TimeDistributed(layers.Dense(64, activation="linear"))(spec)
model = keras.Model(inputs=spec, outputs=[x])


kernel = [[0] * 64 for _ in range(128)]
value = 1
for i in range(128):
    for j in range(64):
        kernel[i][j] = value
        value += 1
        if value > 64:
            value = 1


bias = [0] * 64
value = 1
for i in range(64):
    bias[i] = value
    value += 1
    if value > 64:
        value = 1

kernel = np.array(kernel)
bias = np.array(bias)

model.set_weights([kernel, bias])


input = [[[0] * 128]]

value = 1
for i in range(128):
    input[0][0][i] = value
    value += 1
    if value > 128:
        value = 1

input = np.array(input)

predict = model.predict(input)
print(predict)