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

np.set_printoptions(suppress=True)


initializer = tf.keras.initializers.Zeros()
spec = layers.Input(shape=[None, 64], dtype=np.float32)
x = layers.Bidirectional(layers.GRU(64, recurrent_initializer=initializer, return_sequences=True))(spec)

model = keras.Model(inputs=spec, outputs=[x])

f_kernel = np.array(data.forward_kernel)
f_recurrent_kernel = np.array(data.forward_recurrent_kernel)
f_bias = np.array(data.forward_bias)

b_kernel = np.array(data.backward_kernel)
b_recurrent_kernel = np.array(data.backward_recurrent_kernel)
b_bias = np.array(data.backward_bias)


model.set_weights([
        f_kernel, f_recurrent_kernel, f_bias,
        b_kernel, b_recurrent_kernel, b_bias,
    ])
model.save("model_simple_gru.h5", save_format="h5")


input = np.array(data.input)

predict = model.predict(input)

np.set_printoptions(precision=32)
print(predict)


"""
import numpy as np
import json

jArray = json.dumps(predict.tolist(), indent=4, default=lambda o: o.encode())
open("array.json", "w").write(jArray)
"""
