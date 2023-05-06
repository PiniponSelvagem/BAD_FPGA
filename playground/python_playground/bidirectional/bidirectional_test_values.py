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
INPUT_LAYER_SHAPE = [None, GRU_UNITS]

F_KERNEL = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
F_REC_KERNEL = [
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
]
F_BIAS = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
]

B_KERNEL = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
]
B_REC_KERNEL = [
    [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2], 
    [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
]
B_BIAS = [
    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3], 
    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
]

INPUT = [[[ 0.01, 0.02, 0.03, 0.04 ]]]



#initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=123)
initializer = tf.keras.initializers.Zeros()
spec = layers.Input(shape=INPUT_LAYER_SHAPE, dtype=np.float32)
x = layers.Bidirectional(layers.GRU(GRU_UNITS, recurrent_initializer=initializer, return_sequences=True))(spec)


model = keras.Model(inputs=spec, outputs=[x])


f_kernel = np.array(F_KERNEL)
f_recurrent_kernel = np.array(F_REC_KERNEL)
f_bias = np.array(F_BIAS)

b_kernel = np.array(B_KERNEL)
b_recurrent_kernel = np.array(B_REC_KERNEL)
b_bias = np.array(B_BIAS)


model.set_weights([
        f_kernel, f_recurrent_kernel, f_bias,
        b_kernel, b_recurrent_kernel, b_bias,
    ])
model.save("model_simple_gru.h5", save_format="h5")


input = np.array(INPUT)

predict = model.predict(input)
print(predict)

