import tensorflow as tf
from keras import backend

"""
See tensor value?
https://stackoverflow.com/questions/56262735/how-to-see-tensor-value-of-a-layer-output-in-keras
https://stackoverflow.com/questions/43448029/how-can-i-print-the-values-of-keras-tensors
"""



kernel = []
recurrent_kernel = []
bias = []

input_bias, recurrent_bias = tf.unstack(bias)

def step(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]

    # inputs projected by all gate matrices at once
    matrix_x = backend.dot(cell_inputs, kernel)
    matrix_x = backend.bias_add(matrix_x, input_bias)

    x_z, x_r, x_h = tf.split(matrix_x, 3, axis=1)

    # hidden state projected by all gate matrices at once
    matrix_inner = backend.dot(h_tm1, recurrent_kernel)
    matrix_inner = backend.bias_add(matrix_inner, recurrent_bias)

    recurrent_z, recurrent_r, recurrent_h = tf.split(matrix_inner, 3, axis=1)
    z = tf.sigmoid(x_z + recurrent_z)
    r = tf.sigmoid(x_r + recurrent_r)
    hh = tf.tanh(x_h + r * recurrent_h)

    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    return h, [h]