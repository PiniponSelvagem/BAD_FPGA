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
KERNEL_SHAPE = [1,3]
RECURRENT_KERNEL_SHAPE = [1,3]
BIAS_SHAPE = [2,3]

INPUT_SHAPE = [1,1,1]

#initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=123)
initializer = tf.keras.initializers.Zeros()

spec = layers.Input(shape=INPUT_LAYER_SHAPE, dtype=np.float32)
x = layers.GRU(GRU_UNITS, recurrent_initializer=initializer, return_sequences=True)(spec)
model = keras.Model(inputs=spec, outputs=[x])


kernel = np.zeros(shape=KERNEL_SHAPE)
recurrent_kernel = np.zeros(shape=RECURRENT_KERNEL_SHAPE)
bias = np.ones(shape=BIAS_SHAPE)

model.set_weights([kernel, recurrent_kernel, bias])
model.save("model_simple_gru.h5", save_format="h5")


input = np.zeros(shape=INPUT_SHAPE)

predict = model.predict(input)
print(predict)


"""
################ STEP START ################ None
kernel =  [[0 0 0]]
recurrent_kernel =  [[0 0 0]]
bias =  [[1 1 1] [1 1 1]]
input_bias =  [1 1 1]
recurrent_bias =  [1 1 1]
cell_inputs =  [[0]]
cell_states =  ([[0]],)
------------------------------------------- None
h_tm1 =  [[0]]
matrix_x (dot) =  [[0 0 0]]
matrix_x (bias_add) =  [[1 1 1]]
tf.split =  [[[1]], [[1]], [[1]]]
x_z =  [[1]]
x_r =  [[1]]
x_h =  [[1]]
matrix_inner (dot) =  [[0 0 0]]
matrix_inner (bias_add) =  [[1 1 1]]
recurrent_z =  [[1]]
recurrent_r =  [[1]]
recurrent_h =  [[1]]
x_z + recurrent_z =  [[2]]
z =  [[0.880797088]]
x_r + recurrent_r =  [[2]]
r =  [[0.880797088]]
x_h + r * recurrent_h =  [[1.88079715]]
hh =  [[0.954562962]]
h =  [[0.113786682]]
################ STEP END ################ None
"""


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
