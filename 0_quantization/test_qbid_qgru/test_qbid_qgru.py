
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from tensorflow import keras
from tensorflow.keras import layers

from qkeras.utils import model_quantize
from qkeras import QBidirectional, QGRU


bits = "4"
integer = "1"
integer_gru_recact = "0"
symmetric = "1"

config = {
    "QGRU": {
        "activation": f"quantized_tanh({bits})",
        #"recurrent_activation": ...,
        "state_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
        "kernel_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
        "bias_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
    },
}
#
### model start ###
units = 1
spec = keras.Input(shape=[None, units], dtype=np.float32)
x = QBidirectional(
    QGRU(units,
        activation = f"quantized_tanh({bits})",
        recurrent_activation = f"quantized_relu({bits},{integer_gru_recact},{symmetric})",
        kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
        recurrent_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
        bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
        #state_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
        return_sequences = True
    )
)(spec)
### model end ###
#
_model = keras.Model(inputs=spec, outputs=x)
model = model_quantize(_model, config, bits, transfer_weights=True)



# QGRU does not have "recurrent_bias"

f_kernel = np.array([[-0.875, 0.25, 0.125]])            # [[-0.875 0.21875 0.109375]]
f_recurrent_kernel = np.array([[-0.75, -0.25, 0.75]])   # [[-0.75 -0.21875 0.75]]
f_bias = np.array([0.125, -0.125, 0.875])               # [0 0 1]

b_kernel = np.array([[0, 0, 0]])
b_recurrent_kernel = np.array([[0, 0, 0]])
b_bias = np.array([0, 0, 0])

model.set_weights([
    f_kernel, f_recurrent_kernel, f_bias,
    b_kernel, b_recurrent_kernel, b_bias
])




input_data = np.array([[0.125, -0.125]])
result = model.predict(input_data)
print(result)