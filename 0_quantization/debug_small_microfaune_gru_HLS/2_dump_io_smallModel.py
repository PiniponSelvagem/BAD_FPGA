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
data_type["bits_int"] = 0

# use old rearange
shouldRearrange = False

doPaddingInput = False
START_PADDING = 64  # number of elements to place on the z axis with zeros to achieve padding for HLS

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
X = np.array(
[
    [
        [0.0625, 0.125,  0.1875],
        [0.25,   0.3125, 0.375 ],
        [0.4375, 0.5,    0.5625],
        [0.625,  0.6875, 0.75  ]
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
    if doPaddingInput:
        if layer_name == "input_1" or layer_name == "q_activation":
            # add padding to 1st input layer
            original_shape = out.shape
            # Create a new shape with the same dimensions, except the last dimension is set to 64
            new_shape = list(original_shape)
            new_shape[-1] = START_PADDING
            # Reshape the array to the new shape
            new_array = np.zeros(new_shape)
            new_array[..., 0] = out[..., 0]
            # Convert NumPy array to TensorFlow tensor
            tf_tensor = tf.constant(new_array)
            out = tf_tensor
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
    cutils.saveArray(folder, str(i)+"__"+layer.name, np.array(out), str(i)+"__"+layer.name, data_type, binPositiveOnly=True)
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

