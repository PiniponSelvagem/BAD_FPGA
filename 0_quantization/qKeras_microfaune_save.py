import os
import time
from datetime import timedelta

from microfaune import audio
import tensorflow as tf
from tensorflow import keras
import librosa

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import json
import qkeras
import qkeras.utils as qutils
import qkeras_microfaune_model as qmodel
import keras as K
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

model_dir = "model_quantized"
plots_dir = "plots"
model_subname = "quant_bits411"
bits = "4"
integer = "1"
symmetric = "1"
max_value = "1"
padding = "same"
#model_original, dual_model_original = qmodel.MicrofauneAI.model()
model, dual_model = qmodel.MicrofauneAI(bits, integer, symmetric, max_value, padding).modelQuantized()
dual_model.load_weights(f"{model_dir}/model_weights-{model_subname}.h5")

folder = "model_quantized_weights"

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
for layer in dual_model.layers:
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




def processLayerWeight(i, layerName, weightName, weight):
    shape = weight.shape
    sdim = len(shape)
    data = np.array(weight)
    if any(layerName_c1 in layerName for layerName_c1 in layerName_toChannelFirst):
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
    if layerName_to3d in weightName:
        data = rearange_gru_weights(data)
        if ("kernel" in weightName): # reorder kernel for better memory access / no jumping around when access
            data = data.transpose((1, 2, 0))
    #
    shouldSplit = True
    for name in layerName_split_dim0:
        if name not in weightName:
            shouldSplit = False
            break
    #
    if shouldSplit:
        bias, rbias = np.split(data, 2, axis=0)
        bias = np.squeeze(bias)
        rbias = np.squeeze(rbias)
        biasName = weightName
        rbiasBias = weightName+"_recurrent"
        cutils.saveArray(folder, str(i)+"_"+biasName, bias, biasName, data_type)
        cutils.saveArray(folder, str(i)+"_"+rbiasBias, rbias, rbiasBias, data_type)
    else:
        cutils.saveArray(folder, str(i)+"_"+weightName, data, weightName, data_type)



model_quant = qutils.model_save_quantized_weights(dual_model)

i = 0
for layerName in model_quant:
    layer = model_quant[layerName]
    #for weight in layer:
    weight = layer["weights"]
    if "conv2d" in layerName:
        processLayerWeight(i, layerName, layerName+"_kernel", weight[0])
        processLayerWeight(i, layerName, layerName+"_bias", weight[1])
    if "batch_normalization" in layerName:
        processLayerWeight(i, layerName, layerName+"_gamma", weight[0])
        processLayerWeight(i, layerName, layerName+"_beta", weight[1])
        processLayerWeight(i, layerName, layerName+"_mean", weight[2])
        processLayerWeight(i, layerName, layerName+"_variance", weight[3])
    if "bidirectional" in layerName:
        processLayerWeight(i, layerName, layerName+"_gru_forward_kernel", weight[0])
        processLayerWeight(i, layerName, layerName+"_gru_forward_recurrent_kernel", weight[1])
        processLayerWeight(i, layerName, layerName+"_gru_forward_bias", weight[2])
        processLayerWeight(i, layerName, layerName+"_gru_backward_kernel", weight[3])
        processLayerWeight(i, layerName, layerName+"_gru_backward_recurrent_kernel", weight[4])
        processLayerWeight(i, layerName, layerName+"_gru_backward_bias", weight[5])
    i += 1



# save non quantized layers: time_distributed and time_distributed_1
for layer in dual_model.layers:
    layerName = layer.name
    weight = layer.weights
    if "time_distributed" in layerName:
        processLayerWeight(i, layerName, layerName+"_kernel", weight[0])
        processLayerWeight(i, layerName, layerName+"_bias", weight[1])
        i += 1



#########################################################
end = time.time()
elapsed = end - start


print('Time elapsed: ' + str(timedelta(seconds=elapsed)))

