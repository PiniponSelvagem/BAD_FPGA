import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#########################################################

CUDA_VISIBLE_DEVICES=""     # force use CPU

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

BATCH_SIZE = 2
IN_SIZE    = 8
OUT_SIZE   = 4

spec = layers.Input(shape=[None, IN_SIZE], dtype=np.float32)
x = layers.TimeDistributed(layers.Dense(OUT_SIZE, activation="sigmoid"))(spec)
model = keras.Model(inputs=spec, outputs=[x])


kernel = [[0] * OUT_SIZE for _ in range(IN_SIZE)]
value = 0
for i in range(IN_SIZE):
    for j in range(OUT_SIZE):
        kernel[i][j] = value
        value += 0.01
        #if value > OUT_SIZE:
        #    value = 1

bias = [0] * OUT_SIZE
value = 0
for i in range(OUT_SIZE):
    bias[i] = value
    value += 0.01
    #if value > OUT_SIZE:
    #    value = 1

kernel = np.array(kernel)
bias = np.array(bias)

model.set_weights([kernel, bias])
print("KERNEL:\n"+str(kernel))
print("BIAS:\n"+str(bias))

input = [[[0] * IN_SIZE]]

value = 0
for i in range(IN_SIZE):
    input[0][0][i] = value
    value += 0.01
    #if value > IN_SIZE:
    #    value = 1


input = [[[0] * IN_SIZE] for _ in range(BATCH_SIZE)]

value = 0
for i in range(BATCH_SIZE):
    for j in range(IN_SIZE):
        input[i][0][j] = value
        value += 0.01


input = np.array(input)
print("INPUT:\n"+str(input))

predict = model.predict(input)
print("OUTPUT:\n"+str(predict))