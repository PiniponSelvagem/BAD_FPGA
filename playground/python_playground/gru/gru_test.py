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


#initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=123)
initializer = tf.keras.initializers.Zeros()

spec = layers.Input(shape=[None, 2], dtype=np.float32)
x = layers.GRU(2, recurrent_initializer=initializer, return_sequences=True)(spec)
model = keras.Model(inputs=spec, outputs=[x])


kernel = np.zeros(shape=[2,6])
recurrent_kernel = np.zeros(shape=[2,6])
bias = np.ones(shape=[2,6])

model.set_weights([kernel, recurrent_kernel, bias])
model.save("model_simple_gru.h5", save_format="h5")


input = np.zeros(shape=[1, 1, 2])

predict = model.predict(input)
print(predict)


"""
#testing sigmoid
sig = tf.sigmoid([2.0])
print(sig)

""
for i in range(12):
    sig = tf.sigmoid([float(i)])
    print(i, sig)
""
    
    
#testing tanh
tanh = tf.math.tanh([2.0])
print(tanh)



weight_matrix = tf.Variable(initializer(shape=(2, 6)))
weight_matrix
"""
