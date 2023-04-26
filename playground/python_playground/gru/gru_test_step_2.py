import os
import time

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#########################################################

CUDA_VISIBLE_DEVICES=""     # force use CPU

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

GRU_UNITS = 4
INPUT_LAYER_SHAPE = [None, GRU_UNITS]   # maybe ????
KERNEL = [
    [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2], 
    [2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1],
    [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2],
    [2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3],
]
REC_KERNEL = [
    [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1], 
    [1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0],
    [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1],
    [1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2],
]
BIAS = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
    [0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11],
]

INPUT = [[[ 0.01, 0.02, 0.03, 0.04 ]]]


#initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=123)
initializer = tf.keras.initializers.Zeros()

spec = layers.Input(shape=INPUT_LAYER_SHAPE, dtype=np.float32)
x = layers.GRU(GRU_UNITS, recurrent_initializer=initializer, return_sequences=True)(spec)
model = keras.Model(inputs=spec, outputs=[x])


kernel = np.array(KERNEL)
recurrent_kernel = np.array(REC_KERNEL)
bias = np.array(BIAS)

model.set_weights([kernel, recurrent_kernel, bias])
model.save("model_simple_gru.h5", save_format="h5")


input = np.array(INPUT)

predict = model.predict(input)
print(predict)

