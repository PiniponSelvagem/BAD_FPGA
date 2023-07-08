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



start = time.time()
#########################################################

if not os.path.isdir("model_json"):
    os.makedirs("model_json")

# DUMP get_config
layersConfig = []
for m in model.layers:
    layersConfig.append(m.get_config())

jLayersConfig = json.dumps(layersConfig, indent=4, cls=NumpyEncoder)
open("model_json/dump_layers.json", "w").write(jLayersConfig)


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


layersWeightBias = []
for m in model.layers:
    variables = []
    for v in m.weights:
        shape = v.shape
        sdim = len(shape)
        data = v.numpy()
        # rearrange for channel first, ex: [1, 40, 10, 64] -> [1, 64, 40, 10]
        # NOTE: Currently i have no idea if this is OK for the RNN part
        if sdim == 4:
            [shape[i] for i in (2, 3, 0, 1)]
            data = data.transpose((2, 3, 0, 1))
        elif sdim == 3:
            [shape[i] for i in (0, 2, 1)]
            data = data.transpose((0, 2, 1))
        elif sdim == 2:
            [shape[i] for i in (1, 0)]
            data = data.transpose((1, 0))
        variables.append(ArrayWB(v.name, shape.as_list(), str(v.dtype.name), data.tolist()))
    layer = Layer(
        str(m.name),
        InOut(m.input.shape.as_list(), str(m.input.dtype.name)),   # input
        InOut(m.output.shape.as_list(), str(m.output.dtype.name)), # output
        variables,
    )
    layersWeightBias.append(layer)

jLayersWeightBias = json.dumps(layersWeightBias, indent=4, default=lambda o: o.encode())
open("model_json/dump_weights_bias.json", "w").write(jLayersWeightBias)



# MANUAL QUANTIZATION
#layersConfig -> not quantitized
list_statistics = []

quantValue = 16
def quantitize(value):
    list_statistics.append(str(value))
    list_statistics.append('\n')
    value = value * 2**quantValue
    value = int(value)
    value = value / 2**quantValue
    return value

def quantitizeData(list_):
    for index, item in enumerate(list_):
        if isinstance(item, list):
            quantitizeData(item)
        else:
            list_[index] = quantitize(list_[index])


for layer in layersWeightBias:
    for w in layer.variables:
        index = 0
        for d in w.data:
            if isinstance(d, float):
                w.data[index] = quantitize(w.data[index])
            else:
                quantitizeData(d)
            index += 1


jLayersWeightBiasQuant = json.dumps(layersWeightBias, indent=4, default=lambda o: o.encode())
open("model_json/dump_weights_bias_quant_16.json", "w").write(jLayersWeightBiasQuant)

statistics = ''.join(list_statistics)
open("model_json/dump_weights_bias_statistics.csv", "w").write(statistics)


#########################################################
end = time.time()
elapsed = end - start


print('Time elapsed: ' + str(timedelta(seconds=elapsed)))



m = model.layers[1]
w = map(
    lambda weight: ArrayWB(weight.name, weight.shape.as_list(), str(weight.dtype.name), weight.numpy().tolist()),
    m.weights
)

n = list(w)