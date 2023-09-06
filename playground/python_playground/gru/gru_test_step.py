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


kernel = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
recurrent_kernel = np.array([[-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12]])
bias = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]])

model.set_weights([kernel, recurrent_kernel, bias])
model.save("model_simple_gru.h5", save_format="h5")


input = np.array([[[-2, 1]]])

predict = model.predict(input)
print(predict)


"""

################ STEP START ################ None
kernel =  [[1 2 3 4 5 6] [7 8 9 10 11 12]]
recurrent_kernel =  [[-1 -2 -3 -4 -5 -6] [-7 -8 -9 -10 -11 -12]]
bias =  [[0 0 0 0 0 0] [1 1 1 1 1 1]]
input_bias =  [0 0 0 0 0 0]
recurrent_bias =  [1 1 1 1 1 1]
cell_inputs =  [[20 30]]
cell_states =  ([[0 0]],)
------------------------------------------- None
h_tm1 =  [[0 0]]
matrix_x (dot) =  [[230 280 330 380 430 480]]
matrix_x (bias_add) =  [[230 280 330 380 430 480]]
tf.split =  [[[230 280]], [[330 380]], [[430 480]]]
x_z =  [[230 280]]
x_r =  [[330 380]]
x_h =  [[430 480]]
matrix_inner (dot) =  [[0 0 0 0 -0 -0]]
matrix_inner (bias_add) =  [[1 1 1 1 1 1]]
recurrent_z =  [[1 1]]
recurrent_r =  [[1 1]]
recurrent_h =  [[1 1]]
x_z + recurrent_z =  [[231 281]]
z =  [[1 1]]
x_r + recurrent_r =  [[331 381]]
r =  [[1 1]]
x_h + r * recurrent_h =  [[431 481]]
hh =  [[1 1]]
h =  [[0 0]]
################ STEP END ################ None

"""



# DOT PRODUCT
def K_dot():
    # Define two matrices A and B
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = [[11, 12, 13], [14, 15, 16], [17, 18, 19]]
    #
    # Compute the dot product of A and B
    m, n = len(A), len(B[0])
    C = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
                print("C[",i,"][",j,"]=",C[i][j]," A[",i,"][",k,"]=",A[i][k]," B[",k,"][",j,"]=",B[k][j])
    #
    # Print the result
    for row in C:
        print(row)



# BIAS ADD
def K_biasAdd():
    # Define a 2D array
    x = [[1, 2], [3, 4], [5, 6]]
    #
    # Define a bias vector
    bias = [1, 1]
    #
    # Add the bias vector to the array element-wise using a nested for loop
    y = []
    for i in range(len(x)):
        row = []
        for j in range(len(x[i])):
            row.append(x[i][j] + bias[j])
        y.append(row)
    #
    # Print the result
    for row in y:
        print(row)


