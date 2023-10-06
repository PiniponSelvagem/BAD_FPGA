
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from tensorflow import keras
import tensorflow.keras.backend as K
from qkeras.utils import model_quantize
from qkeras import QGRU, quantized_tanh, quantized_relu

bits = "4"
integer = "1"
integer_gru_recact = "0"
symmetric = "1"

### model start ###
units = 2
spec = keras.Input(shape=[None, 1], dtype=np.float32)
x = QGRU(units,
        activation = f"quantized_tanh({bits})",
        recurrent_activation = f"quantized_relu({bits},{integer_gru_recact},{symmetric})",
        kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
        recurrent_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
        bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
        #state_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
        return_sequences = True
)(spec)
### model end ###
#
_model = keras.Model(inputs=spec, outputs=x)
model = model_quantize(_model, None, bits, transfer_weights=True)



# QGRU does not have "recurrent_bias"

#kernel = np.array([[-0.875, 0.25, 0.125]])            # [[-0.875 0.21875 0.109375]]
#recurrent_kernel = np.array([[-0.75, -0.25, 0.75]])   # [[-0.75 -0.21875 0.75]]
#bias = np.array([0.125, -0.125, 0.875])               # [0 0 1]

kernel = np.array([[-0.875, 0.25, 0.125,    -0.125, 0.375, -0.25]])
recurrent_kernel = np.array([
    [-0.125, -0.25, 0.75,    -0.375,  0.125,  0.5],
    [ 0.125,  0,    0.25,     0.375, -0.125, -0.5],
])
bias = np.array([-0.875, -0.125, -0.5,     0.875, -0.125, 0.375])


model.set_weights([
    kernel, recurrent_kernel, bias
])




input_data = np.array([[0.125, -0.125]])
result = model.predict(input_data)
print(result)








#############################################
units = 2
quantized_kernel = np.array([[-0.875, 0.21875, 0.109375,    -0.109375, 0.375, -0.21875]])   # np.array([[-0.875, 0.25, 0.125,    -0.125, 0.375, -0.25]])
quantized_recurrent = np.array([[-0.109375, -0.21875, 0.75,    -0.375, 0.109375, 0.4375], [0.109375, 0, 0.25,    0.375, -0.109375, -0.4375]])   # np.array([[-0.125, -0.25, 0.75,    -0.375,  0.125,  0.5], [ 0.125,  0,    0.25,     0.375, -0.125, -0.5]])
input_bias = np.array([-1, 0, -0.5,    1, 0, 0.5])  # np.array([-0.875, -0.125, -0.5,     0.875, -0.125, 0.375])
inputs = np.array([0.125, -0.125])
h_tm1 = np.array([[0, 0]])
#####
#
def activation(value):
    return K.eval(quantized_tanh(4)(value))

def recurrent_activation(value):
    return K.eval(quantized_relu(4,0,1)(value))

def gru_cell(input, h_tm1):
    ####################################################
    # The "split", example:
    # xpto = [ 1, 2, 3, 4, 5, 6 ]
    # z = xpto[:, :units])              ---> z = [1, 2] 
    # r = xpto[:, units:units * 2])     ---> r = [3, 4]
    # h = xpto[:, units * 2:])          ---> h = [5, 6]
    ####################################################
    print("############### STEP START ###############")
    print("quantized_kernel =", quantized_kernel)
    print("quantized_recurrent (kernel) =", quantized_recurrent)
    print("input_bias =", input_bias)
    print("inputs =", inputs)
    print("h_tm1 =", h_tm1)
    print("------------------------------------------")
    inputs_z = input
    inputs_r = input
    inputs_h = input
    print("inputs_z =", inputs_z)
    print("inputs_r =", inputs_r)
    print("inputs_h =", inputs_h)
    #
    x_z = np.dot(inputs_z, quantized_kernel[:, :units])
    x_r = np.dot(inputs_r, quantized_kernel[:, units:units * 2])
    x_h = np.dot(inputs_h, quantized_kernel[:, units * 2:])
    print("x_z (dot) =", x_z)
    print("x_r (dot) =", x_r)
    print("x_h (dot) =", x_h)
    #
    x_z = x_z + input_bias[:units]
    x_r = x_r + input_bias[units: units * 2]
    x_h = x_h + input_bias[units * 2:]
    print("x_z (bias_add) =", x_z)
    print("x_r (bias_add) =", x_r)
    print("x_h (bias_add) =", x_h)
    #
    h_tm1_z = h_tm1
    h_tm1_r = h_tm1
    h_tm1_h = h_tm1
    print("h_tm1_z =", h_tm1_z)
    print("h_tm1_r =", h_tm1_r)
    print("h_tm1_h =", h_tm1_h)
    #
    recurrent_z = np.dot(h_tm1_z, quantized_recurrent[:, :units])
    recurrent_r = np.dot(h_tm1_r, quantized_recurrent[:, units:units * 2])
    print("recurrent_z =", recurrent_z)
    print("recurrent_r =", recurrent_r)
    #
    z = recurrent_activation(x_z + recurrent_z) # recurrent_activation: quantized_relu
    r = recurrent_activation(x_r + recurrent_r) # recurrent_activation: quantized_relu
    print("z =", z)
    print("r =", r)
    #
    recurrent_h = np.dot(r * h_tm1_h, quantized_recurrent[:, units * 2:])
    print("recurrent_h =", recurrent_h)
    #
    hh = activation(x_h + recurrent_h) # activation: quantized_tahn
    print("hh =", hh)
    #
    h = z * h_tm1 + (1 - z) * hh
    print("h =", h)
    print("################ STEP END ################")
    h_tm1 = h   # save output as state
    return [h, h_tm1]


out0, h_tm1 = gru_cell(np.array(inputs[0]), h_tm1)
out1, h_tm1 = gru_cell(np.array(inputs[1]), h_tm1)







"""
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
h
"""
