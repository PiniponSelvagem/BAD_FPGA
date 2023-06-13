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

def forward_gru(input):
    print("#### FORWARD ####")
    initializer = tf.keras.initializers.Zeros()
    #
    spec = layers.Input(shape=[None, 64], dtype=np.float32)
    x = layers.GRU(64, recurrent_initializer=initializer, return_sequences=True)(spec)
    model = keras.Model(inputs=spec, outputs=[x])
    #
    kernel = np.array(data.forward_kernel)
    recurrent_kernel = np.array(data.forward_recurrent_kernel)
    bias = np.array(data.forward_bias)
    #
    model.set_weights([kernel, recurrent_kernel, bias])
    #model.save("model_simple_gru.h5", save_format="h5")
    #
    #input = np.array(data.input)
    predict = model.predict(input)
    #
    predict_flat = predict.flatten().tolist()
    outexpc_flat = [item for sublist in data.output_expected for item in sublist]
    #
    warn = False
    for i in range(len(predict_flat)):
        if predict_flat[i] != outexpc_flat[i]:
            print(f"Warn at {i}, output: {predict_flat[i]} | expected: {outexpc_flat[i]}")
            warn = True
    #
    if (warn == False):
        print("Predict values are equal to expected output values.")
    #
    return predict_flat

def backward_gru(input):
    print("#### BACKWARD ####")
    initializer = tf.keras.initializers.Zeros()
    #
    spec = layers.Input(shape=[None, 64], dtype=np.float32)
    x = layers.GRU(64, recurrent_initializer=initializer, return_sequences=True)(spec)
    model = keras.Model(inputs=spec, outputs=[x])
    #
    kernel = np.array(data.backward_kernel)
    recurrent_kernel = np.array(data.backward_recurrent_kernel)
    bias = np.array(data.backward_bias)
    #
    model.set_weights([kernel, recurrent_kernel, bias])
    #model.save("model_simple_gru.h5", save_format="h5")
    #
    #input = np.array(data.input)
    predict = model.predict(input)
    #
    predict_flat = predict.flatten().tolist()
    outexpc_flat = [item for sublist in data.output_expected for item in sublist]
    #
    warn = False
    offset = 64
    for i in range(len(predict_flat)):
        i_offset = i + offset
        if predict_flat[i] != outexpc_flat[i_offset]:
            print(f"Warn at {i}, output: {predict_flat[i]} | expected: {outexpc_flat[i_offset]}")
            warn = True
    #
    if (warn == False):
        print("Predict values are equal to expected output values.")
    #
    return predict_flat


forward_input  = np.array(data.input)
backward_input = np.array(data.input)

forward_predict  = forward_gru(forward_input)
backward_predict = backward_gru(backward_input)
