import os
import real_data as data

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
x = layers.Conv2D(64, (3, 3), padding="same", activation=None)(spec)
model = keras.Model(inputs=spec, outputs=[x])


kernel = np.array(data.kernel)
bias = np.array(data.bias)

model.set_weights([kernel, bias])
model.save("model_simple_conv2d.h5", save_format="h5")

predict = model.predict(data.input)

print(" #### OUTPUT ####")
for i in range(40):
    print(predict[0][0][i][0])
