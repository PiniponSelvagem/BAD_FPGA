import tensorflow as tf

import data

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#########################################################

CUDA_VISIBLE_DEVICES=""     # force use CPU

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

IN_SIZE = 128
OUT_SIZE= 64

spec = layers.Input(shape=[None, IN_SIZE], dtype=np.float32)
x = layers.TimeDistributed(layers.Dense(OUT_SIZE, activation="sigmoid"))(spec)
model = keras.Model(inputs=spec, outputs=[x])


kernel = np.array(data.kernel)
bias = np.array(data.bias)
model.set_weights([kernel, bias])

input = np.array(data.input)

predict = model.predict(input)
print("OUTPUT:\n"+str(predict))
