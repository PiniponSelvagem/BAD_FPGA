
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from tensorflow import keras
from tensorflow.keras import layers

from qkeras.utils import model_quantize

bits = "4"
integer = "1"
symmetric = "1"

config = {
    "QGRU": {
        "activation": f"quantized_tanh({bits})",
        #"recurrent_activation": ...,
        "state_quantizer": f"quantized_bits({bits},0,{symmetric})",
        "kernel_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
        "bias_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
    },
}
#
### model start ###
units = 1
spec = keras.Input(shape=[None, units], dtype=np.float32)
x = layers.GRU(units, return_sequences=True)(spec)
### model end ###
#
_model = keras.Model(inputs=spec, outputs=x)
model = model_quantize(_model, config, bits, transfer_weights=True)



"""
kernel = np.array([[0.5, -0.5, 0.5]])
recurrent_kernel = np.array([[0.5, -0.5, 0.5]])
bias = np.array([[0.5, -0.5, 0.5], [-0.5, 0.5, -0.5]])

model.set_weights([kernel, recurrent_kernel, bias])
"""


start_value = -5
end_value = 5
step = 0.001

i = 0
for value in np.arange(start_value, end_value + step, step):
    kernel = np.random.uniform(-1.0, 1.0, size=(1, 3))
    recurrent_kernel = np.random.uniform(-1.0, 1.0, size=(1, 3))
    bias = np.random.uniform(-1.0, 1.0, size=(2, 3))
    #
    model.set_weights([kernel, recurrent_kernel, bias])
    #
    print("ID:", i)
    print("VALUE:", value)
    input_data = np.array([[[value], [1], [1]]])
    result = model.predict(input_data)
    i += 1


