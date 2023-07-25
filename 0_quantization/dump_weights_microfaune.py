import os
import time
from datetime import timedelta

from microfaune.detection import RNNDetector
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import json

import cutils

# Extend the JSONEncoder class
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

detector = RNNDetector()
model = detector.model


folder = "model_dump"

layerName_toSave = [
    "conv2d",
    "batch_normalization",
    "re_lu",
    "max_pooling2d",
    "bidirectional",
    "time_distributed",
]
layerName_toChannelFirst = [
    "conv2d",
    "batch_normalization",
    "re_lu",
    "max_pooling2d",
    "time_distributed"
]
layerName_to3d = "gru"
layerName_split_dim0 = [
    "gru",
    "bias",
]

layerName_toMerge = [
    # hardcoded to only work with conv2d and batch_normalization
    "conv2d",
    "batch_normalization"
]


data_type = {}
data_type["name"] = "float"
"""
data_type["name"] = "ap_fixed"
data_type["bits_total"] = 16
data_type["bits_int"] = 7
"""

start = time.time()
#########################################################

if not os.path.isdir(folder):
    os.makedirs(folder)

# DUMP get_config
layersConfig = []
for m in model.layers:
    layersConfig.append(m.get_config())

jLayersConfig = json.dumps(layersConfig, indent=4, cls=NumpyEncoder)
open(folder+"/dump_layers.json", "w").write(jLayersConfig)


def rearange_gru_weights(data):
    rows, cols = data.shape
    split = 3
    jump = cols // split  # Jump value calculated as size of the last dimension divided by 'split'
    #
    new_cols = jump * split
    new_array = data[:, :new_cols].reshape(rows, split, jump).swapaxes(1, 2)
    #
    return new_array



# DUMP weights and bias
class InOut:
    def __init__(self, shape, type):
        self.shape = shape
        self.type = type
    def encode(self):
        return self.__dict__

class ArrayWB:
    def __init__(self, name, shape, type, data):
        self.name = name
        self.shape = shape
        self.type = type
        self.data = data
    def encode(self):
        return self.__dict__

class Layer:
    def __init__(self, name, input, output, variables):
        self.name = name
        self.input = input
        self.output = output
        self.variables = variables
    def encode(self):
        return self.__dict__


layers_toMerge = []
i = 0
for m in model.layers:
    if any(layerName_save in m.name for layerName_save in layerName_toSave):
        variables = []
        for v in m.weights:
            shape = v.shape
            sdim = len(shape)
            data = v.numpy()
            if any(layerName_c1 in m.name for layerName_c1 in layerName_toChannelFirst):
                # change to channel first
                if sdim == 4:
                    [shape[i] for i in (3, 2, 0, 1)]
                    data = data.transpose((3, 2, 0, 1)) #fc
                elif sdim == 3:
                    [shape[i] for i in (0, 2, 1)]
                    data = data.transpose((0, 2, 1))
                elif sdim == 2:
                    [shape[i] for i in (1, 0)]
                    data = data.transpose((1, 0))
            arrayName = v.name.replace('/', '_').replace('.', '_').replace(':', '_')
            if layerName_to3d in v.name:
                data = rearange_gru_weights(data)
                if ("kernel" in v.name): # reorder kernel for better memory access / no jumping around when access
                    data = data.transpose((1, 2, 0))
            #
            for name in layerName_toMerge:
                if name in v.name:
                    layerData = {
                        "layerID": 0,
                        "layerName": m.name,
                        "layerFullName": arrayName,
                        "layerWeightName": "/".join(v.name.split("/")[1:]).replace(':', '_'),
                        "data": data
                    }
                    layers_toMerge.append(layerData)
                    break
            #
            shouldSplit = True
            for name in layerName_split_dim0:
                if name not in v.name:
                    shouldSplit = False
                    break
            if shouldSplit:
                bias, rbias = np.split(data, 2, axis=0)
                bias = np.squeeze(bias)
                rbias = np.squeeze(rbias)
                biasName = arrayName
                rbiasBias = arrayName+"_recurrent"
                cutils.saveArray(folder, str(i)+"_"+biasName, bias, biasName, data_type)
                cutils.saveArray(folder, str(i)+"_"+rbiasBias, rbias, rbiasBias, data_type)
            #
            else:
                #data = [-35.51, 2.5, -0.5, 0.125, -0.125, 0.05, 0.625, -0.625, 0.02698972076177597]
                #data = np.array(data)
                #if "conv2d_bias_0" in arrayName:
                cutils.saveArray(folder, str(i)+"_"+arrayName, data, arrayName, data_type)
    i += 1


EPSILON = 0.0001
merged_layers = []
for x in range(0, len(layers_toMerge), 6):
    conv2d_kernel_full = layers_toMerge[x]
    conv2d_bias_full = layers_toMerge[x + 1]
    #
    conv2d_kernel = conv2d_kernel_full["data"]
    conv2d_bias = layers_toMerge[x + 1]["data"]
    #
    bn_gamma = layers_toMerge[x + 2]["data"]
    bn_beta = layers_toMerge[x + 3]["data"]
    bn_moving_mean = layers_toMerge[x + 4]["data"]
    bn_moving_variance = layers_toMerge[x + 5]["data"]
    #
    # Merge the Conv2D weights with BatchNormalization weights and bias terms
    merged_kernel = np.zeros_like(conv2d_kernel)
    merged_bias = np.zeros_like(conv2d_bias)
    for i in range(conv2d_kernel.shape[0]):
        for j in range(conv2d_kernel.shape[1]):
            merged_kernel[i, j] = conv2d_kernel[i, j] / np.sqrt(bn_moving_variance[i] + EPSILON)
        merged_bias[i] = (conv2d_bias[i] - bn_moving_mean[i]) / np.sqrt(bn_moving_variance[i] + EPSILON) * bn_gamma[i] + bn_beta[i]
    #
    # Create a new merged layer and append it to the list
    merged_layer = {
        "layerName": "conv2d",
        "layerFullName": "merged_"+conv2d_kernel_full["layerFullName"],
        "layerWeightName": conv2d_kernel_full["layerWeightName"],
        "data": merged_kernel
    }
    merged_layers.append(merged_layer)
    #
    merged_layer_bias = {
        "layerName": "conv2d",
        "layerFullName": "merged_"+conv2d_bias_full["layerFullName"],
        "layerWeightName": conv2d_bias_full["layerWeightName"],
        "data": merged_bias
    }
    merged_layers.append(merged_layer_bias)

for layer in merged_layers:
    cutils.saveArray(folder, layer["layerFullName"], layer["data"], layer["layerFullName"], data_type)



#########################################################
end = time.time()
elapsed = end - start


print('Time elapsed: ' + str(timedelta(seconds=elapsed)))






""" # playing around with conv2d and batch_normalization merging
#input
input = 1.1

#conv2d
kernel = 0.5
bias = 1

#batch_normalization
EPSILON = 0.001
gamma = 0.1
beta = 0.2
movingvariance = 0.3
movingmean = 0.4

#conv2d
output_conv2d = (input * kernel) + bias
output_conv2d
#batch_normalization
normalized = (output_conv2d - movingmean) / np.sqrt(movingvariance + EPSILON)
output = gamma * normalized + beta
output

# merged weights: conv2d <-> batch_normalization
W_merged = kernel / np.sqrt(movingvariance + EPSILON) * gamma
B_merged = (bias - movingmean) / np.sqrt(movingvariance) * gamma + beta

# using merged weights
inout = (input * W_merged) + B_merged
inout
"""
