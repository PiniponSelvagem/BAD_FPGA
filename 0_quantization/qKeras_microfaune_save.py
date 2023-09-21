import time
from datetime import timedelta

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import json
import qkeras.utils as qutils
import qkeras_microfaune_model as qmodel
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

from model_config.config_test import ModelConfig      # model_test
#from model_config.config_0 import ModelConfig       # model_quant_411
#from model_config.config_1 import ModelConfig       # model_quant__conv-po2-81_gru-po2-81_bnorm-811

model_folder = ModelConfig.folder
model_name = ModelConfig.name
#model_original, dual_model_original = qmodel.MicrofauneAI.model()
model, dual_model = qmodel.MicrofauneAI(ModelConfig).modelQuantized()
dual_model.load_weights(f"{model_folder}/{model_name}.h5")

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


save_merged = False
data_type = {}
data_type["name"] = "float"
"""
data_type["name"] = "ap_fixed"
data_type["bits_total"] = 8
data_type["bits_int"] = 1
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


jConfig = json.dumps(dual_model.get_config(), indent=4, cls=NumpyEncoder)
open(folder+"/dump_config.json", "w").write(jConfig)


def rearange_gru_weights(data):
    rows, cols = data.shape
    split = 3
    jump = cols // split  # Jump value calculated as size of the last dimension divided by 'split'
    #
    new_cols = jump * split
    new_array = data[:, :new_cols].reshape(rows, split, jump).swapaxes(1, 2)
    #
    return new_array

def rearange_gru_scales(data):
    if data.size > 1:
        filters = data.shape[0]
        split = 3
        jump = filters // split  # Jump value calculated as size of the last dimension divided by 'split'
        #
        new_array = data.reshape(split, jump).swapaxes(0, 1)
        #
        return new_array
    else:
        return data



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




def processLayer(layerName, weightName, weight, isScale=False):
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
        if isScale:
            data = rearange_gru_scales(data)
            # TODO: check if the code below is actually required for scales
            #if ("kernel" in weightName): # reorder kernel for better memory access / no jumping around when access
            #    data = data.transpose((0, 1))
        else:
            data = rearange_gru_weights(data)
            if ("kernel" in weightName): # reorder kernel for better memory access / no jumping around when access
                data = data.transpose((1, 2, 0))
    #
    shouldSplit = True
    for name in layerName_split_dim0:
        if name not in weightName:
            shouldSplit = False
            break
    if isScale:
        shouldSplit = False
    #
    if shouldSplit:
        bias, rbias = np.split(data, 2, axis=0)
        bias = np.squeeze(bias)
        rbias = np.squeeze(rbias)
        biasName = weightName
        rbiasBias = weightName+"_recurrent"
        cutils.saveArray(folder, biasName, bias, biasName, data_type)
        cutils.saveArray(folder, rbiasBias, rbias, rbiasBias, data_type)
    else:
        cutils.saveArray(folder, weightName, data, weightName, data_type)



model_quant = qutils.model_save_quantized_weights(dual_model)

"""
import keras.backend as K
for layer in dual_model.layers:
    try:
        if layer.get_quantizers():
            q_w_pairs = zip(layer.get_quantizers(), layer.get_weights())
        for _, (quantizer, weight) in enumerate(q_w_pairs):
            qweight = K.eval(quantizer(weight))
            qscale = 1.0
            if hasattr(quantizer, "scale") and quantizer.scale is not None:
                qscale = K.eval(quantizer.scale)
            print(f"quantized weight ({layer.name}) (scale: {qscale}, {np.array(qscale).shape})")
            print(qweight)
    except AttributeError:
        print("warning, the weight is not quantized in the layer ", layer.name)
"""






# save weights "scale"
import keras.backend as K
for layer in dual_model.layers:
    try:
        if layer.get_quantizers():
            q_w_pairs = zip(layer.get_quantizers(), layer.get_weights())
        for _, (quantizer, weight) in enumerate(q_w_pairs):
            qscale = 1.0
            if hasattr(quantizer, "scale") and quantizer.scale is not None:
                qscale = K.eval(quantizer.scale)
            print(f"{layer.name}, scale: {np.array(qscale).flatten().shape}")
    except AttributeError:
        print("warning, the weight is not quantized in the layer ", layer.name)




"""
i = 0
for l in dual_model.layers:
    print(str(i)+", "+str(l.name))
    i+=1
"""
layers_idx = {
    'conv2d': 1,
    'conv2d_1': 4,
    'conv2d_2': 8,
    'conv2d_3': 11,
    'conv2d_4': 15,
    'conv2d_5': 18,
    'batch_normalization': 2,
    'batch_normalization_1': 5,
    'batch_normalization_2': 9,
    'batch_normalization_3': 12,
    'batch_normalization_4': 16,
    'batch_normalization_5': 19,
    'bidirectional': 23,
    'bidirectional_1': 24
}

def getQuantizeScale(layerName, weight_idx):
    layer = dual_model.layers[layers_idx.get(layerName)]
    if layer.get_quantizers():
        qscale = layer.get_quantizers()[weight_idx].scale
        if isinstance(qscale, float):
            qscale = [qscale]
    else:
        qscale = [1.0]
    return np.array(qscale).flatten()


print("save_merged == "+str(save_merged))
"""
processLayer("conv2d", "conv2d_kernel", model_quant["conv2d"]["weights"][0])
processLayer("conv2d", "conv2d_kernel_scale", getQuantizeScale("conv2d", 0))
processLayer("conv2d", "conv2d_bias_scale", getQuantizeScale("conv2d", 1), isScale=True)
"""
for layerName in model_quant:
    layer = model_quant[layerName]
    #for weight in layer:
    weight = layer["weights"]
    if "conv2d" in layerName:
        if save_merged:
            processLayer(layerName, layerName+"_kernel", weight[0])
            processLayer(layerName, layerName+"_bias", layer["fused_bias"])
        else:
            processLayer(layerName, layerName+"_kernel", weight[0])
            processLayer(layerName, layerName+"_bias", weight[1])
            #
            processLayer(layerName, layerName+"_kernel_scale", getQuantizeScale(layerName, 0), isScale=True)
            processLayer(layerName, layerName+"_bias_scale", getQuantizeScale(layerName, 1), isScale=True)
    if "batch_normalization" in layerName:
        processLayer(layerName, layerName+"_gamma", weight[0])
        processLayer(layerName, layerName+"_beta", weight[1])
        processLayer(layerName, layerName+"_mean", weight[2])
        processLayer(layerName, layerName+"_variance", weight[3])
        #
        processLayer(layerName, layerName+"_gamma_scale", getQuantizeScale(layerName, 0), isScale=True)
        processLayer(layerName, layerName+"_beta_scale", getQuantizeScale(layerName, 1), isScale=True)
        processLayer(layerName, layerName+"_mean_scale", getQuantizeScale(layerName, 2), isScale=True)
        processLayer(layerName, layerName+"_variance_scale", getQuantizeScale(layerName, 3), isScale=True)
    if "bidirectional" in layerName:
        processLayer(layerName, layerName+"_gru_forward_kernel", weight[0])
        processLayer(layerName, layerName+"_gru_forward_recurrent_kernel", weight[1])
        processLayer(layerName, layerName+"_gru_forward_bias", weight[2])
        processLayer(layerName, layerName+"_gru_backward_kernel", weight[3])
        processLayer(layerName, layerName+"_gru_backward_recurrent_kernel", weight[4])
        processLayer(layerName, layerName+"_gru_backward_bias", weight[5])
        #
        processLayer(layerName, layerName+"_gru_forward_kernel_scale", getQuantizeScale(layerName, 0), isScale=True)
        processLayer(layerName, layerName+"_gru_forward_recurrent_kernel_scale", getQuantizeScale(layerName, 1), isScale=True)
        processLayer(layerName, layerName+"_gru_forward_bias_scale", getQuantizeScale(layerName, 2), isScale=True)
        processLayer(layerName, layerName+"_gru_backward_kernel_scale", getQuantizeScale(layerName, 3), isScale=True)
        processLayer(layerName, layerName+"_gru_backward_recurrent_kernel_scale", getQuantizeScale(layerName, 4), isScale=True)
        processLayer(layerName, layerName+"_gru_backward_bias_scale", getQuantizeScale(layerName, 5), isScale=True)

# save non quantized layers: time_distributed and time_distributed_1
for layer in dual_model.layers:
    layerName = layer.name
    weight = layer.weights
    if "time_distributed" in layerName:
        processLayer(layerName, layerName+"_kernel", weight[0])
        processLayer(layerName, layerName+"_bias", weight[1])




#########################################################
end = time.time()
elapsed = end - start


print('Time elapsed: ' + str(timedelta(seconds=elapsed)))
