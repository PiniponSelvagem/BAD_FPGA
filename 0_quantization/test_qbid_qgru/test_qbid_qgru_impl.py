
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from tensorflow import keras
import tensorflow.keras.backend as K
from qkeras.utils import model_quantize
from qkeras import QBidirectional, QGRU, quantized_tanh, quantized_relu

bits = "4"
integer = "1"
integer_gru_recact = "0"
symmetric = "1"

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
model = model_quantize(_model, None, bits, transfer_weights=True)



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




#############################################
units = 1
#quantized_kernel = np.array([[-0.875, 0.25, 0.125]])    # f_kernel
#quantized_recurrent = np.array([[-0.75, -0.25, 0.75]])  # f_reccurent_kernel
#input_bias = np.array([0.125, -0.125, 0.875])           # f_bias
quantized_kernel = np.array([[-0.875, 0.21875, 0.109375]])  # f_kernel (from print)
quantized_recurrent = np.array([[-0.75, -0.21875, 0.75]])   # f_reccurent_kernel (from print)
input_bias = np.array([0, 0, 1])                            # f_bias (from print)
inputs = np.array([[0.125]])                            # input
h_tm1 = np.array([[0.875]]) # state
#####
#
def activation(value):
    return K.eval(quantized_tanh(4)(value))

def recurrent_activation(value):
    return K.eval(quantized_relu(4,0,1)(value))


inputs_z = inputs
inputs_r = inputs
inputs_h = inputs

x_z = np.dot(inputs_z, quantized_kernel[:, :units])
x_r = np.dot(inputs_r, quantized_kernel[:, units:units * 2])
x_h = np.dot(inputs_h, quantized_kernel[:, units * 2:])

x_z = x_z + input_bias[:units]
x_r = x_r + input_bias[units: units * 2]
x_h = x_h + input_bias[units * 2:]

h_tm1_z = h_tm1
h_tm1_r = h_tm1
h_tm1_h = h_tm1

recurrent_z = np.dot(h_tm1_z, quantized_recurrent[:, :units])
recurrent_r = np.dot(h_tm1_r, quantized_recurrent[:, units:units * 2])

z = recurrent_activation(x_z + recurrent_z) # recurrent_activation: quantized_relu
r = recurrent_activation(x_r + recurrent_r) # recurrent_activation: quantized_relu

recurrent_h = np.dot(r * h_tm1_h, quantized_recurrent[:, units * 2:])

hh = activation(x_h + recurrent_h) # activation: quantized_tahn
h = z * h_tm1 + (1 - z) * hh





import numpy as np

bias = np.array([
    10, 20, 11, 21, 12, 22
])

units = 2
a = bias[:units]
b = bias[units: units * 2]
c = bias[units * 2:]



num_rows = len(bias) // 3

reshaped_bias = bias.reshape(num_rows, -1, order='F')
reshaped_bias
