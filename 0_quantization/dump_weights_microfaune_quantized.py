import os
import time
from datetime import timedelta

import tensorflow as tf
import keras.backend as K

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import json

import cutils
import qkeras_microfaune_model as qmodel

#from model_config.config_test import ModelConfig            # model_test
#from model_config.config_0 import ModelConfig               # model_quant_411
from model_config.config_0_noQuantState import ModelConfig  # model_quant_411_noQuantState
#from model_config.config_0_qconvbnorm import ModelConfig    # model_quant_411_qconvbnorm
#from model_config.config_1 import ModelConfig               # model_quant__conv-po2-81_gru-po2-81_bnorm-811

models_folder = ModelConfig.folder
model_name = ModelConfig.name

model_folder = ModelConfig.folder
model_name = ModelConfig.name
model, dual_model = qmodel.MicrofauneAI(ModelConfig).modelQuantized()

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

model.load_weights(f"{models_folder}/{model_name}.h5")

folder = "model_dump_quantized"

layerName_toSave = [
    "conv2d",
    "batch_normalization",
    "bidirectional",
    "time_distributed",
]
layerName_toChannelFirst = [
    "conv2d",
    "batch_normalization",
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
data_type["bits_total"] = 30
data_type["bits_int"] = 6
"""

start = time.time()
#########################################################

cutils.createFolderIfNotExists(folder)

# DUMP get_config
layersConfig = []
for layer in model.layers:
    layersConfig.append(layer.get_config())

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




def processLayerWeight(layer, weight_full, weight, layers_toMerge: list):
    shape = weight.shape
    sdim = len(shape)
    data = np.array(weight) #.numpy()
    if any(layerName_c1 in layer.name for layerName_c1 in layerName_toChannelFirst):
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
    #
    arrayName = weight_full.name.replace('/', '_').replace('.', '_').replace(':', '_')
    if layerName_to3d in weight_full.name:
        data = rearange_gru_weights(data)
        if ("kernel" in weight_full.name): # reorder kernel for better memory access / no jumping around when access
            data = data.transpose((1, 2, 0))
    #
    for name in layerName_toMerge:
        if name in weight_full.name:
            layerData = {
                "layerID": 0,
                "layerName": layer.name,
                "layerFullName": arrayName,
                "layerWeightName": "/".join(weight_full.name.split("/")[1:]).replace(':', '_'),
                "data": data
            }
            layers_toMerge.append(layerData)
            break
    #
    shouldSplit = True
    for name in layerName_split_dim0:
        if name not in weight_full.name:
            shouldSplit = False
            break
    #
    if shouldSplit:
        bias, rbias = np.split(data, 2, axis=0)
        bias = np.squeeze(bias)
        rbias = np.squeeze(rbias)
        biasName = arrayName
        rbiasBias = arrayName+"_recurrent"
        cutils.saveArray(folder, str(i)+"_"+biasName, bias, biasName, data_type)
        cutils.saveArray(folder, str(i)+"_"+rbiasBias, rbias, rbiasBias, data_type)
    else:
        cutils.saveArray(folder, str(i)+"_"+arrayName, data, arrayName, data_type)




layers_toMerge = []
i = 0
for layer in model.layers:
    if any(layerName_save in layer.name for layerName_save in layerName_toSave):
        try:
            if layer.get_quantizers():
                q_gw_w_pairs = [(quantizer, gweight, weight) for quantizer, gweight, weight in zip(layer.get_quantizers(), layer.get_weights(), layer.weights)]
                for _, (quantizer, gweight, weight) in enumerate(q_gw_w_pairs):
                    qweight = K.eval(quantizer(gweight))
                    #print(weight.name)
                    processLayerWeight(layer, weight, qweight, layers_toMerge)
        except AttributeError:
            print("WARNING: weight is not quantized in layer", layer.name)
            for weight in layer.weights:
                processLayerWeight(layer, weight, weight, layers_toMerge)
    i += 1




EPSILON = 0.001
merged_layers = []
print("WARNING: CONV2D and BATCHNORMALIZATION merging resulting weights are not quantized!!!"+"\n         Merging never was correctly calculated, so not giving much attention to it for now.")
# Merging hasnt been adjusted / tested for qKeras quantization
for layer in range(0, len(layers_toMerge), 6):
    conv2d_kernel_full = layers_toMerge[layer]
    conv2d_bias_full = layers_toMerge[layer + 1]
    #
    conv2d_kernel = conv2d_kernel_full["data"]
    conv2d_bias = layers_toMerge[layer + 1]["data"]
    #
    bn_gamma = layers_toMerge[layer + 2]["data"]
    bn_beta = layers_toMerge[layer + 3]["data"]
    bn_moving_mean = layers_toMerge[layer + 4]["data"]
    bn_moving_variance = layers_toMerge[layer + 5]["data"]
    #
    # Merge the Conv2D weights with BatchNormalization weights and bias terms
    merged_kernel = np.zeros_like(conv2d_kernel)
    merged_bias = np.zeros_like(conv2d_bias)
    for f in range(conv2d_kernel.shape[0]):
        for c in range(conv2d_kernel.shape[1]):
            for x in range(conv2d_kernel.shape[2]):
                for y in range(conv2d_kernel.shape[3]):
                    merged_kernel[f, c, x, y] = conv2d_kernel[f, c, x, y] / np.sqrt(bn_moving_variance[f] + EPSILON) * bn_gamma[f]
        merged_bias[f] = (conv2d_bias[f] - bn_moving_mean[f]) / np.sqrt(bn_moving_variance[f] + EPSILON) * bn_gamma[f] + bn_beta[f]
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
import numpy as np
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

"""
import numpy as np
#input
input = 0.4088122546672821

#conv2d
kernel = 0.029623493552207947
bias = 0.02698972076177597

#batch_normalization
EPSILON = 0.001
gamma = 0.9056835770606995
beta = -0.043935634195804596
movingvariance = 3.146245944662951e-05
movingmean = 0.02926195226609707

#conv2d
output_conv2d = (input * kernel) + bias
output_conv2d
#batch_normalization
normalized = (output_conv2d - movingmean) / np.sqrt(movingvariance + EPSILON)
output = gamma * normalized + beta
output

# merged weights: conv2d <-> batch_normalization
W_merged = kernel / np.sqrt(movingvariance + EPSILON) * gamma
W_merged
B_merged = (bias - movingmean) / np.sqrt(movingvariance + EPSILON) * gamma + beta
B_merged

# using merged weights
inout = (input * W_merged) + B_merged
inout
"""