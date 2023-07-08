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

INPUT_LAYER_SHAPE = [None, 1]
GRU_UNITS = 1

F_KERNEL_SHAPE  = [1,3]
F_RECURRENT_KERNEL_SHAPE = [1,3]
F_BIAS_SHAPE = [2,3]

B_KERNEL_SHAPE = [1,3]
B_RECURRENT_KERNEL_SHAPE = [1,3]
B_BIAS_SHAPE = [2,3]

INPUT_SHAPE = [1,1,1]


#initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=123)
initializer = tf.keras.initializers.Zeros()
spec = layers.Input(shape=INPUT_LAYER_SHAPE, dtype=np.float32)
x = layers.Bidirectional(layers.GRU(GRU_UNITS, recurrent_initializer=initializer, return_sequences=True))(spec)


model = keras.Model(inputs=spec, outputs=[x])


f_kernel = np.zeros(shape=F_KERNEL_SHAPE)
f_recurrent_kernel = np.zeros(shape=F_RECURRENT_KERNEL_SHAPE)
f_bias = np.ones(shape=F_BIAS_SHAPE)

b_kernel = np.zeros(shape=B_KERNEL_SHAPE)
b_recurrent_kernel = np.zeros(shape=B_RECURRENT_KERNEL_SHAPE)
b_bias = np.ones(shape=B_BIAS_SHAPE)


model.set_weights([
        f_kernel, f_recurrent_kernel, f_bias,
        b_kernel, b_recurrent_kernel, b_bias,
    ])
#model.save("model_simple_gru.h5", save_format="h5")


input = np.zeros(shape=INPUT_SHAPE)

predict = model.predict(input)
print(predict)

