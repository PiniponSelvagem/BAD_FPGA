import os

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#########################################################

CUDA_VISIBLE_DEVICES=""     # force use CPU

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


initializer = tf.keras.initializers.Zeros()

spec = layers.Input(shape=[None, 4, 2], dtype=np.float32)
x = layers.Conv2D(2, (3, 3), padding="same", activation=None)(spec)
model = keras.Model(inputs=spec, outputs=[x])

weights = [
    [ [[ 0, 0 ], [ 0, 0 ]], [[ 0, 0 ], [ 0, 0 ]], [[ 0, 0 ], [ 0, 0 ]] ],
    [ [[ 0, 0 ], [ 0, 0 ]], [[ 2, 0 ], [ 4, 0 ]], [[ 0, 0 ], [ 0, 0 ]] ],
    [ [[ 0, 0 ], [ 0, 0 ]], [[ 0, 0 ], [ 0, 0 ]], [[ 0, 0 ], [ 0, 0 ]] ]
]

bias = [ 1, 0 ]

kernel = np.array(weights)
bias = np.array(bias)

model.set_weights([kernel, bias])
#model.save("model_simple_conv2d.h5", save_format="h5")

# bias + (weight[0] * input[0]) + (weight[1] * input[1])
# (1) + (2*2) + (1*4)

input = [
    [
        [ [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ] ],
        [ [ 0, 0 ], [ 2, 1 ], [ 0, 0 ], [ 0, 0 ] ],
        [ [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ] ],
        [ [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ] ]
    ]
]
input = np.array(input)
predict = model.predict(input)


print(" #### INPUT ####")
print(input.shape)
print("--- 0 ---")
for j in range(input.shape[1]):
    for i in range(input.shape[2]):
        print(str("{:4}".format(input[0][j][i][0]))+" ", end='')
    print()
print("--- 1 ---")
for j in range(input.shape[1]):
    for i in range(input.shape[2]):
        print(str("{:4}".format(input[0][j][i][1]))+" ", end='')
    print()
print()
print(" #### OUTPUT ####")
print(predict.shape)
print("--- 0 ---")
for j in range(predict.shape[1]):
    for i in range(predict.shape[2]):
        print(str("{:4}".format(predict[0][j][i][0]))+" ", end='')
    print()
print("--- 1 ---")
for j in range(predict.shape[1]):
    for i in range(predict.shape[2]):
        print(str("{:4}".format(predict[0][j][i][1]))+" ", end='')
    print()