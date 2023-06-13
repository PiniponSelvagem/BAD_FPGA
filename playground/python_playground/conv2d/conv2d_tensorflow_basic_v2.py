import os
import data

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

spec = layers.Input(shape=[None, 40, 1], dtype=np.float32)
x = layers.Conv2D(1, (3, 3), padding="same", activation=None)(spec)
model = keras.Model(inputs=spec, outputs=[x])


weights_debug = [
        [
            [[ 0 ]],
            [[ 1 ]],
            [[ 0 ]]
        ],
        [
            [[ 1 ]],
            [[ 2 ]],
            [[ 1 ]]
        ],
        [
            [[ 0 ]],
            [[ 1 ]],
            [[ 0 ]]
        ]
    ]
bias_debug = [ 1 ]

kernel = np.array(weights_debug)
bias = np.array(bias_debug)

model.set_weights([kernel, bias])
#model.save("model_simple_conv2d.h5", save_format="h5")

input_3 = [
    [
        [
            [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]
        ],
        [
            [0],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]
        ],
        [
            [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]
        ]
    ]
]
input_2 = [
    [
        [
            [0],
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0]
        ],
        [
            [0],
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0]
        ]
    ]
]
input_1 = [
    [
        [
            [0],
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0]
        ]
    ]
]

input = np.array(input_3)
predict = model.predict(input)


print(" #### INPUT ####")
print(input.shape)
for j in range(input.shape[1]):
    for i in range(40):
        print(str("{:4}".format(input[0][j][i][0]))+" ", end='')
    print()
print()
print(" #### OUTPUT ####")
print(predict.shape)
for j in range(predict.shape[1]):
    for i in range(40):
        print(str("{:4}".format(predict[0][j][i][0]))+" ", end='')
    print()

