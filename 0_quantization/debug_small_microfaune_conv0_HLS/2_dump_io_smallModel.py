import os
import time
from datetime import timedelta

import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import json

from smallMicrofauneModel import SmallModelMicrofauneAI
import cutils

data_type = {}
"""
data_type["name"] = "float"
"""
data_type["name"] = "ap_fixed"
data_type["bits_total"] = 4
data_type["bits_int"] = 1

# use old rearange
shouldRearrange = False



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


model_dir = "model_debug"       #ModelConfig.folder
model_name = "smallModel"       #ModelConfig.name
model = SmallModelMicrofauneAI().modelQuantized()

datasets_dir = '../../datasets'
model.load_weights(f"{model_dir}/{model_name}.h5")

folder = "model_debug_dump_io_quantized"
filepath = folder+"/smallModel"
ext = ".json"

if not os.path.isdir(folder):
    os.makedirs(folder)


start = time.time()
#########################################################

#### predicting with the model ####
print("Predicting...")
INPUT = 0
if INPUT == 0:
    X = np.array(
    [
        [
            [[0.125],[0.25]],
            [[0.375],[0.5 ]],
            [[0.625],[0.75]],
        ]
    ])
else:
    X = np.array(
    [
        [
            [   [1],    [0.5],  [1]     ],
            [   [0.25], [0.25], [0.5]   ],
            [   [0.75], [1],    [-0.25] ],
            [   [-1],   [0],    [0.5]   ]
        ]
    ])
result = model.predict(X)
print(" -> result =", result)



intermediate_models = []
layers_names = []
for layer in model.layers:
    layers_names.append(layer.name)
    intermediate_model = keras.Model(inputs=model.input, outputs=layer.output)
    intermediate_models.append(intermediate_model)

class Layer:
    def __init__(self, name, shape, type, output):
        self.name = name
        self.shape = shape
        self.type = type
        self.output = output
    def encode(self):
        return self.__dict__

def getOutputOfLayer(layer_name):
    features = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(name=layer_name).output,
    )
    return features(np.array(X))

layersIO = []
layersIOflat = []
i = 0
for layer_name in layers_names:
    print(layer_name)
    out = getOutputOfLayer(layer_name)
    sdim = len(out.shape)
    if shouldRearrange:
        # rearrange for channel first, ex: [1, 40, 10, 64] -> [1, 64, 40, 10]
        if sdim == 4:
            out = tf.transpose(out, perm=[0, 3, 1, 2])
        elif sdim == 3:
            out = tf.transpose(out, perm=[0, 2, 1])
        #elif sdim == 2:
        #    out = tf.transpose(out, perm=[1, 0])
    #
    layer = Layer(
        str(layer_name),
        out.shape.as_list(),
        out.dtype.name,
        out.numpy().tolist()
    )
    #
    # 20231120 - hotfix to create binary outputs of every layer
    cutils.saveArray(folder, str(i)+"__"+layer.name, out, str(i)+"__"+layer.name, data_type)
    #
    layersIO.append(layer)
    jLayer = json.dumps(layer, indent=4, default=lambda o: o.encode())
    open(filepath+"__"+str(i)+"__"+layer_name.replace(".", "_")+ext, "w").write(jLayer)
    layerflat = Layer(
        str(layer_name),
        out.shape.as_list(),
        out.dtype.name,
        out.numpy().ravel().tolist()
    )
    layersIOflat.append(layerflat)
    jLayerFlat = json.dumps(layerflat, indent=4, default=lambda o: o.encode())
    open(filepath+"_flat__"+str(i)+"__"+layer_name.replace(".", "_")+ext, "w").write(jLayerFlat)
    i += 1

"""
jLayersIO = json.dumps(layersIO, indent=4, default=lambda o: o.encode())
open(filepath+ext, "w").write(jLayersIO)
jLayersIOflat = json.dumps(layersIOflat, indent=4, default=lambda o: o.encode())
open(filepath+"_flat"+ext, "w").write(jLayersIOflat)
"""


#########################################################
end = time.time()
elapsed = end - start


print('Time elapsed: ' + str(timedelta(seconds=elapsed)))

