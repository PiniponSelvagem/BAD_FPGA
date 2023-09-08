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

audiofile = "no_bird_79266"
audiofolder = "audio_samples"
audiofilePath_w_ext = audiofolder+"/"+audiofile+".wav"
folder = "features_inout"
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

#### original code, but here for convinience ####
fs, data = audio.load_wav(audiofilePath_w_ext)
X = compute_features([data])
input = np.array(X)

#### predicting with the model ####
scores, local_scores = model.predict(input)
print(audiofile+" -> score="+str(scores[0][0]))



start = time.time()
#########################################################

data_type = {}
data_type["name"] = "float"

# save input
cutils.saveArray(folder, audiofile+"_input", input.reshape((431,40)), "input", data_type)

# save outputs
cutils.saveArray(folder, audiofile+"_output_global", scores.reshape((1)), "output_global", data_type)
cutils.saveArray(folder, audiofile+"_output_local", local_scores.reshape((431,1)), "output_local", data_type)

#########################################################
end = time.time()
elapsed = end - start


print('Time elapsed: ' + str(timedelta(seconds=elapsed)))
