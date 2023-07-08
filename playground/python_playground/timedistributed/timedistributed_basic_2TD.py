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
x = layers.TimeDistributed(layers.Dense(2, activation="sigmoid"))(spec)
local_pred = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
model = keras.Model(inputs=spec, outputs=[x])


kernel = [
        [ 1, 1 ],
        [ 0, 0 ],
    ]
bias = [ 0, 0 ]

kernel = np.array(kernel)
bias = np.array(bias)

model.set_weights([kernel, bias])

input = [
    [
        [[1, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
    ],
    [
        [[1, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
    ]
]

input = np.array(input)

predict = model.predict(input)
print(predict)


"""
`Dense` implements the operation:
`output = activation(dot(input, kernel) + bias)`
where `activation` is the element-wise activation function
passed as the `activation` argument, `kernel` is a weights matrix
created by the layer, and `bias` is a bias vector created by the layer
(only applicable if `use_bias` is `True`). These are all attributes of
`Dense`.
"""