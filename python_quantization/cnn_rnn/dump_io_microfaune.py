import os
import time
from datetime import timedelta

from microfaune.detection import RNNDetector
from microfaune import audio
import tensorflow as tf
from tensorflow import keras
import librosa

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


audiofile = "1_a_bird"
audiofile_w_ext = audiofile+".wav"
folder = "dump_io"
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
      data, sr=fs, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
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

#### original code, but here for convinience ####
fs, data = audio.load_wav(audiofile_w_ext)
X = compute_features([data])

#### predicting with the model ####
scores, local_scores = model.predict(np.array(X))
print(audiofile+" -> score="+str(scores[0][0]))


print("Dumping input and outputs of each layer to: "+filepath+ext)



start = time.time()
#########################################################

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
