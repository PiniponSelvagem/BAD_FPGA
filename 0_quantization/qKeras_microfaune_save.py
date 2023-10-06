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

#from model_config.config_test import ModelConfig            # model_test
#from model_config.config_0 import ModelConfig               # model_quant_411
#from model_config.config_0_noQuantState import ModelConfig  # model_quant_411_noQuantState
from model_config.config_0_qconvbnorm import ModelConfig    # model_quant_411_qconvbnorm
#from model_config.config_1 import ModelConfig               # model_quant__conv-po2-81_gru-po2-81_bnorm-811

model_folder = ModelConfig.folder
model_name = ModelConfig.name
#model_original, dual_model_original = qmodel.MicrofauneAI.model()
model, dual_model = qmodel.MicrofauneAI(ModelConfig).modelQuantized()
dual_model.load_weights(f"{model_folder}/{model_name}.h5")

folder = "model_quantized_weights"

layerName_toChannelFirst = [
    #"q_conv2d_batchnorm"
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
data_type["bits_total"] = 8
data_type["bits_int"] = 1
"""

start = time.time()
#########################################################
isManualQmodel = False

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
    if isManualQmodel and len(data.shape)==1:   # it is bias of QGru
        cols = data.shape[0]
        split = 3
        jump = cols // split
        #
        new_cols = jump * split
        new_array = data[:new_cols].reshape(split, -1, order='F')       # fill the new shape by column, elements in the same column of the original array will be adjacent in memory
    else:
        rows, cols = data.shape
        split = 3
        jump = cols // split  # Jump value calculated as size of the last dimension divided by 'split'
        #
        new_cols = jump * split
        new_array = data[:, :new_cols].reshape(rows, split, jump).swapaxes(1, 2)
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
    if isScale or isManualQmodel:
        # scale is only one dim always
        # QGRU bias does not require split because there is no reccurrent bias
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
    print("'"+str(l.name)+"': "+str(i)+",")
    i+=1
"""
if "q_conv2d_batchnorm" in dual_model.layers[1].name:
    isManualQmodel = True
    layers_idx = {
        'q_conv2d_batchnorm': 1,
        'q_activation': 2,
        'q_conv2d_batchnorm_1': 3,
        'q_activation_1': 4,
        'max_pooling2d': 5,
        'q_conv2d_batchnorm_2': 6,
        'q_activation_2': 7,
        'q_conv2d_batchnorm_3': 8,
        'q_activation_3': 9,
        'max_pooling2d_1': 10,
        'q_conv2d_batchnorm_4': 11,
        'q_activation_4': 12,
        'q_conv2d_batchnorm_5': 13,
        'q_activation_5': 14,
        'max_pooling2d_2': 15,
        'q_bidirectional': 17,
        'q_bidirectional_1': 18
    }
else:
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


def getQuantizeScale(layerName, idx):
    layer = dual_model.layers[layers_idx.get(layerName)]
    if layer.get_quantizers():
        quant = layer.get_quantizers()[idx]
        if quant is not None:
            qscale = quant.scale
            if isinstance(qscale, float):
                qscale = [qscale]
        else:
            print(f"INFO: {layerName}, quantizer[{idx}] does not have a quantizer.")
            qscale = [1.0]
    else:
        qscale = [1.0]
    return np.array(qscale).flatten()


"""
processLayer("conv2d", "conv2d_kernel", model_quant["conv2d"]["weights"][0])
processLayer("conv2d", "conv2d_kernel_scale", getQuantizeScale("conv2d", 0))
processLayer("conv2d", "conv2d_bias_scale", getQuantizeScale("conv2d", 1), isScale=True)
"""
for layerName in model_quant:
    print(layerName)
    layer = model_quant[layerName]
    #for weight in layer:
    weight = layer["weights"]
    if "conv2d" in layerName:
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
        processLayer(layerName, layerName+"_gru_forward_state_scale", getQuantizeScale(layerName, 3), isScale=True)
        processLayer(layerName, layerName+"_gru_backward_kernel_scale", getQuantizeScale(layerName, 4), isScale=True)
        processLayer(layerName, layerName+"_gru_backward_recurrent_kernel_scale", getQuantizeScale(layerName, 5), isScale=True)
        processLayer(layerName, layerName+"_gru_backward_bias_scale", getQuantizeScale(layerName, 6), isScale=True)
        processLayer(layerName, layerName+"_gru_backward_state_scale", getQuantizeScale(layerName, 7), isScale=True)


"""
Bidirectional GRU
qs = l.get_quantizers()
    # Forward maybe
q = qs[0]   -> kernel_quantizer
q = qs[1]   -> recurrent_quantizer
q = qs[2]   -> bias_quantizer
q = qs[3]   -> state_quantizer
    # Backward maybe
q = qs[4]   -> kernel_quantizer
q = qs[5]   -> recurrent_quantizer
q = qs[6]   -> bias_quantizer
q = qs[7]   -> state_quantizer
"""

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










############################################################################
# NOTE: placing in line 82 'print(data.shape)'
# for microfaune bidirectional gives:
# bidirectional
# (64, 192)
# (64, 192)
# (2, 192)
# (64, 192)
# (64, 192)
# (2, 192)
#
# for qconv2dbatchnorm variant gives:
# q_bidirectional
# (64, 192)
# (64, 192)
# (192,)
# Traceback (most recent call last):
#   File "/mnt/e/Rodrigo/ISEL/2_Mestrado/2-ANO_1-sem/TFM/BAD_FPGA/0_quantization/qKeras_microfaune_save.py", line 319, in <module>
#     processLayer(layerName, layerName+"_gru_forward_bias", weight[2])
#   File "/mnt/e/Rodrigo/ISEL/2_Mestrado/2-ANO_1-sem/TFM/BAD_FPGA/0_quantization/qKeras_microfaune_save.py", line 158, in processLayer
#     data = rearange_gru_weights(data)
#   File "/mnt/e/Rodrigo/ISEL/2_Mestrado/2-ANO_1-sem/TFM/BAD_FPGA/0_quantization/qKeras_microfaune_save.py", line 83, in rearange_gru_weights
#     rows, cols = data.shape
# ValueError: not enough values to unpack (expected 2, got 1)
#
# 
# This was with a super fast training
# 
############################################################################
"""




layer = model_quant["q_bidirectional"]

processLayer(layerName, layerName+"_gru_forward_kernel", weight[0])
processLayer(layerName, layerName+"_gru_forward_recurrent_kernel", weight[1])





def rearange_gru_weights(data):
    print("DATA:"+str(data))
    rows, cols = data.shape
    split = 3
    jump = cols // split  # Jump value calculated as size of the last dimension divided by 'split'
    #
    new_cols = jump * split
    new_array = data[:, :new_cols].reshape(rows, split, jump).swapaxes(1, 2)
    #
    return new_array


w = weight[2]

shape = w.shape
sdim = len(shape)
data = np.array(w)


processLayer(layerName, layerName+"_gru_forward_bias", weight[2])







processLayer(layerName, layerName+"_gru_backward_kernel", weight[3])
processLayer(layerName, layerName+"_gru_backward_recurrent_kernel", weight[4])
processLayer(layerName, layerName+"_gru_backward_bias", weight[5])







processLayer(layerName, layerName+"_gru_forward_kernel_scale", getQuantizeScale(layerName, 0), isScale=True)
processLayer(layerName, layerName+"_gru_forward_recurrent_kernel_scale", getQuantizeScale(layerName, 1), isScale=True)
processLayer(layerName, layerName+"_gru_forward_bias_scale", getQuantizeScale(layerName, 2), isScale=True)
processLayer(layerName, layerName+"_gru_forward_state_scale", getQuantizeScale(layerName, 3), isScale=True)
processLayer(layerName, layerName+"_gru_backward_kernel_scale", getQuantizeScale(layerName, 4), isScale=True)
processLayer(layerName, layerName+"_gru_backward_recurrent_kernel_scale", getQuantizeScale(layerName, 5), isScale=True)
processLayer(layerName, layerName+"_gru_backward_bias_scale", getQuantizeScale(layerName, 6), isScale=True)
processLayer(layerName, layerName+"_gru_backward_state_scale", getQuantizeScale(layerName, 7), isScale=True)

"""


