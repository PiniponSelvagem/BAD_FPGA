
# True will use CPU instead of GPU when predicting and also taking execution time.
FORCE_CPU = True

##################################################

import time
from datetime import timedelta

import numpy as np
import random
random.seed(666)

from microfaune.detection import RNNDetector
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if FORCE_CPU:
    tf.config.set_visible_devices([], 'GPU')


dataset = "ff1010bird"
basePath = "/mnt/e/Rodrigo/ISEL/2_Mestrado/2-ANO_1-sem/TFM"


# Load original model
from qkeras_microfaune_model_1cell import MicrofauneAI
microfaune = MicrofauneAI()
_, modelOrig = microfaune.modelOriginal()

##################################################

import microfaune.audio as audio
from scipy.io import wavfile
from scipy import signal
import librosa

#### original code, but here for convinience ####
def load_wav(path, decimate=None):
    fs, data = wavfile.read(path)
    data = data.astype(np.float32)
    if decimate is not None:
        data = signal.decimate(data, decimate)
        fs /= decimate
    return fs, data

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
def predict(X):
    scores = []
    local_scores = []
    for x in X:
        s, local_s = modelOrig.predict(x[np.newaxis, ...])
        scores.append(s[0])
        local_scores.append(local_s.flatten())
    scores = np.array(s)
    return scores, local_scores

#### original code, but here for convinience ####
def predict_on_wav(wav_file):
    fs, data = audio.load_wav(wav_file)
    X = compute_features([data])
    scores, local_scores = predict(np.array(X))
    return scores[0], local_scores[0]




audioDir = basePath+"/files_audio/"+dataset+"/wav/"
metadata_path = basePath+"/files_audio_metadata/"+dataset+"_metadata.txt"
class_file = np.loadtxt(metadata_path, delimiter=',',dtype='str') # file with metadata in format: ID_OF_WAV,hasBird

# making sure the model is in cache
maxRange = 10
for i in range(maxRange):
    fs, data = load_wav(audioDir+class_file[i][0]+".wav")
    X = compute_features([data])
    scoreOrig, _  = modelOrig.predict(np.array(X))


fs, data = load_wav(audioDir+class_file[maxRange][0]+".wav")
X = compute_features([data])

start = time.time()
#########################################################
scoreOrig, _  = modelOrig.predict(np.array(X))
#########################################################
end = time.time()
elapsed = end - start


print('Predict time: ' + str(timedelta(seconds=elapsed)))

elapsed_milliseconds = elapsed * 1000  # Convert seconds to milliseconds
print('Predict time: {:.2f} milliseconds'.format(elapsed_milliseconds))
