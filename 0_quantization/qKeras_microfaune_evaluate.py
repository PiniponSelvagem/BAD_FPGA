
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from qkeras import *

import qkeras_microfaune_model as qmodel

# config_0_noQuantState | config_0_qconv2dbnorm
wav_file = "audio_samples/bird_50124.wav" # data_bird_0     # 0.98303944 | 0.9866601
#wav_file = "audio_samples/bird_52046.wav" # data_bird_1     # 0.8675719  | 0.901361
#wav_file = "audio_samples/bird_16835.wav" # data_bird_2     # 0.921088   | 0.9615345
#wav_file = "audio_samples/bird_80705.wav" # data_bird_3     # 0.9579106  | 0.9538768

#wav_file = "audio_samples/no_bird_50678.wav" # data_no_bird_0       # 0.04347346 | 0.019658929
#wav_file = "audio_samples/no_bird_51034.wav" # data_no_bird_1       # 0.05368349 | 0.04402802
#wav_file = "audio_samples/no_bird_1931.wav"  # data_no_bird_2       # 0.07247392 | 0.029453337
#wav_file = "audio_samples/no_bird_79266.wav" # data_no_bird_3       # 0.11154108 | 0.121614665


#from model_config.config_test import ModelConfig            # model_test
#from model_config.config_0 import ModelConfig               # model_quant_411
from model_config.config_0_noQuantState import ModelConfig  # model_quant_411_noQuantState
#from model_config.config_0_qconvbnorm import ModelConfig    # model_quant_411_qconvbnorm
#from model_config.config_1 import ModelConfig               # model_quant__conv-po2-81_gru-po2-81_bnorm-811

model_folder = ModelConfig.folder
model_name = ModelConfig.name
model, dual_model = qmodel.MicrofauneAI(ModelConfig).modelQuantized()
"""
model = model_original
dual_model = dual_model_original
"""



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

