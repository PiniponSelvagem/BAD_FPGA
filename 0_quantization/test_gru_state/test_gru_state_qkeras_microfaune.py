
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



wav_file = "../audio_samples/bird_50124.wav" # data_bird_0     # 0.98303944
#wav_file = "audio_samples/bird_52046.wav" # data_bird_1     # 0.8675719
#wav_file = "audio_samples/bird_16835.wav" # data_bird_2     # 0.921088
#wav_file = "audio_samples/bird_80705.wav" # data_bird_3     # 0.9579106

#wav_file = "audio_samples/no_bird_50678.wav" # data_no_bird_0       # 0.04347346
#wav_file = "audio_samples/no_bird_51034.wav" # data_no_bird_1       # 0.05368349
#wav_file = "audio_samples/no_bird_1931.wav"  # data_no_bird_2       # 0.07247392
#wav_file = "audio_samples/no_bird_79266.wav" # data_no_bird_3       # 0.11154108

import sys
sys.path.append('..')

from qkeras import *
import qkeras_microfaune_model as qmodel

#from model_config.config_test import ModelConfig
from model_config.config_0_quantState_401 import ModelConfig

model_folder = "../"+ModelConfig.folder
model_name = ModelConfig.name
model, dual_model = qmodel.MicrofauneAI(ModelConfig).modelQuantized()




from scipy.io import wavfile
from scipy import signal
import librosa
def load_wav(path, decimate=None):
    fs, data = wavfile.read(path)
    data = data.astype(np.float32)
    if decimate is not None:
        data = signal.decimate(data, decimate)
        fs /= decimate
    return fs, data

def create_spec(data, fs, n_mels=32, n_fft=2048, hop_len=1024):
    # Calculate spectrogram
    S = librosa.feature.melspectrogram(
      y=data, sr=fs, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
    S = S.astype(np.float32)
    # Convert power to dB
    S = librosa.power_to_db(S)
    return S

def compute_features(audio_signals):
    X = []
    for data in audio_signals:
        x = create_spec(data, fs=44100, n_mels=40, n_fft=2048,
                        hop_len=1024).transpose()
        X.append(x[..., np.newaxis].astype(np.float32)/255)
    return X



fs, data = load_wav(wav_file)
X = compute_features([data])


dual_model.load_weights(f"{model_folder}/{model_name}.h5")
#wav_files = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(datasets_dir, "*/wav/*.wav"))}
scores, local_scores = dual_model.predict(np.array(X))

print(f"Local: {local_scores}")
print(f"Global: {scores}")



