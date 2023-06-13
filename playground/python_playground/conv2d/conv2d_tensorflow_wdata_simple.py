import os
import real_data_1 as data
n_filter = 1

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
x = layers.Conv2D(n_filter, (3, 3), padding="same", activation=None)(spec)
model = keras.Model(inputs=spec, outputs=[x])


kernel = np.array(data.kernel)
bias = np.array(data.bias)

model.set_weights([kernel, bias])
model.save("model_simple_conv2d.h5", save_format="h5")

input = data.input_all
predict = model.predict(input)
print(predict.shape)

print(" #### OUTPUT ####")
MAX_TO_SHOW = 3
for j in range(min(predict.shape[1], MAX_TO_SHOW)):
    for i in range(40):
        print(predict[0][j][i][0])
    print("####")
