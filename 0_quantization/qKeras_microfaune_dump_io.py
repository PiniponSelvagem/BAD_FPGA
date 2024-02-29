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
import qkeras_microfaune_model as qmodel

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

doPaddingInput = True
START_PADDING = 32   # 64/2 because half a byte    # number of padding to place on the z axis with zeros to achieve padding for HLS


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
#from model_config.config_0_qconvbnorm import ModelConfig    # model_quant_411_qconvbnorm
#from model_config.config_0_qconvbnorm__input_relu import ModelConfig    # model_quant_411_qconvbnorm__input_relu
#
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits32 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits32
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits16 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits16
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits8 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits8
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits4 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits4
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits2 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits2
from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1
#
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUnoBias import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUnoBias
#from model_config.config_1 import ModelConfig               # model_quant__conv-po2-81_gru-po2-81_bnorm-811


datasets_dir = '../../datasets'
model_dir = ModelConfig.folder
model_name = ModelConfig.name
plots_dir = "plots"
epochs = ModelConfig.epochs
steps_per_epoch = ModelConfig.steps_per_epoch
#model_original, dual_model_original = qmodel.MicrofauneAI.model()
model, dual_model = qmodel.MicrofauneAI(ModelConfig).modelQuantized()
dual_model.load_weights(f"{model_dir}/{model_name}.h5")

audioIDX = 0
if audioIDX == 0: audiofile = "bird_50124"
if audioIDX == 1: audiofile = "bird_52046"
if audioIDX == 2: audiofile = "bird_16835"
if audioIDX == 3: audiofile = "bird_80705"
if audioIDX == 4: audiofile = "no_bird_50678"
if audioIDX == 5: audiofile = "no_bird_51034"
if audioIDX == 6: audiofile = "no_bird_1931"
if audioIDX == 7: audiofile = "no_bird_79266"
audiofilepath = "audio_samples/"+audiofile+".wav"
folder = "dump_io_quantized"
filepath = folder+"/"+audiofile
ext = ".json"

if not os.path.isdir(folder):
    os.makedirs(folder)

"""
# original code
global_score, local_score = detector.predict_on_wav(audiofile_w_ext)
print(audiofile+" -> score="+str(global_score))
"""

#### original code, but here for convinience ####
def create_spec(data, fs, n_mels=32, n_fft=2048, hop_len=1024):
    # Calculate spectrogram
    S = librosa.feature.melspectrogram(
      y=data, sr=fs, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
    S = S.astype(np.float32)
    # Convert power to dB
    S = librosa.power_to_db(S)
    return S

#### original code, but here for convinience ####
def compute_features(audio_signals):
    X = []
    for data in audio_signals:
        x = create_spec(data, fs=44100, n_mels=40, n_fft=2048,
                        hop_len=1024).transpose()
        X.append(x[..., np.newaxis].astype(np.float32)/255)
    return X

start = time.time()
#########################################################

#### original code, but here for convinience ####
print("Loading wav... If it gets locked here, try to run as administrator.")
fs, data = audio.load_wav(audiofilepath)
X = compute_features([data])

#### predicting with the model ####
print("Predicting...")
scores, local_scores = dual_model.predict(np.array(X))
print(audiofile+" -> score="+str(scores[0][0]))


print("Dumping input and outputs of each layer to: "+audiofile)




intermediate_models = []
layers_names = []
for layer in dual_model.layers:
    layers_names.append(layer.name)
    intermediate_model = keras.Model(inputs=dual_model.input, outputs=layer.output)
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
        inputs=dual_model.inputs,
        outputs=dual_model.get_layer(name=layer_name).output,
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
        if layer_name == "q_activation":        # or layer_name == "input_1
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
    if ("bidirectional" in layer.name) or ("time_distributed" in layer.name) or ("reduce_max_1" in layer.name):
        float_data_type = data_type.copy()
        float_data_type["name"] = "float"
        print(" > INFO: Using", float_data_type["name"], "data_type for this layer.")
        cutils.saveArray(folder, str(i)+"__"+layer.name, np.array(out), str(i)+"__"+layer.name, float_data_type, binPositiveOnly=True)
    else:
        if (("q_activation" == layer.name)):
            cutils.saveArray(folder, str(i)+"__"+layer.name, np.array(out), str(i)+"__"+layer.name, data_type, binPositiveOnly=True, nPacket=1)
        else:
            if ("reduce_max" in layer.name):
                rmax_data_type = {}
                rmax_data_type["name"] = "ap_fixed"
                rmax_data_type["bits_total"] = 8
                rmax_data_type["bits_int"] = 1
                cutils.saveArray(folder, str(i)+"__"+layer.name, np.array(out), str(i)+"__"+layer.name, rmax_data_type, binPositiveOnly=True, nPacket=1)
            else:
                cutils.saveArray(folder, str(i)+"__"+layer.name, np.array(out), str(i)+"__"+layer.name, data_type, binPositiveOnly=True, saveBin=False)

    #
    """
    #### JSON output disabled ####
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
    """
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

