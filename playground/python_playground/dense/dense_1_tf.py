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

spec = layers.Input(shape=[None, 4], dtype=np.float32)
x = layers.Dense(1, activation="linear")(spec)
model = keras.Model(inputs=spec, outputs=[x])

kernel = [ [1], [2], [3], [4] ]
bias = [ 1 ]

kernel = np.array(kernel)
bias = np.array(bias)

model.set_weights([kernel, bias])

input = [ [ 6, 7, 8, 9 ] ]

input = np.array(input)

predict = model.predict(input)
print(predict)


# output = (i[0]*k[0] + i[1]*k[1] + i[2]*k[2] + i[3]*k[3]) + bias[0]