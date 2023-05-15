from microfaune import audio
import numpy as np
import librosa
from microfaune.detection import RNNDetector

import utils
from utils import Layer, Stats


def create_spec(data, fs, n_mels=32, n_fft=2048, hop_len=1024):
    # Calculate spectrogram
    S = librosa.feature.melspectrogram(
      data, sr=fs, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
    S = S.astype(np.float32)
    # Convert power to dB
    S = librosa.power_to_db(S)
    return S

def compute_features(audio_signals):
    X = []
    for data in audio_signals:
        x = create_spec(data, fs=44100, n_mels=40, n_fft=2048, hop_len=1024).transpose()
        X.append(x[..., np.newaxis].astype(np.float32)/255)
    return X


def add_or_get_layer(layers, name, shape):
    for layer in layers:
        if layer.name == name:
            return layer
    new_layer = Layer(name, shape, None)
    return new_layer

def calculate_statistics(th, audiofolder, wav_names, results):
    model = RNNDetector().model
    layers = []
    i = 0
    n_wavs = len(wav_names)
    for wav in wav_names:
        percent_complete = (i / n_wavs) * 100
        percent_str = f"{percent_complete:.4f}%"
        print(str(th) + " -> "+ str(percent_str) + " ("+str(i)+" of "+str(n_wavs)+")")
        _, data = audio.load_wav(audiofolder+wav)
        X = compute_features([data])
        
        for mlayer in model.layers:
            out = utils.getOutputOfLayer(mlayer.name, model, X)
            layer = add_or_get_layer(layers, mlayer.name, out.shape.as_list())
            if layer.stats == None:
                stats = Stats(
                    min = np.min(out),
                    max = np.max(out),
                    range = np.ptp(out),
                    mean = [np.mean(out)],
                    median = [np.median(out)],
                    std = [np.std(out)],
                    var = [np.var(out)],
                )
                layer.stats = stats
                layers.append(layer)
            else:
                layer.stats.min = np.min([np.min(layer.stats.min), np.min(out)])
                layer.stats.max = np.max([np.max(layer.stats.max), np.max(out)])
                layer.stats.range = layer.stats.max - layer.stats.min
                layer.stats.mean.append(np.mean(out))
                layer.stats.median.append(np.median(out))
                layer.stats.std.append(np.std(out)) 
                layer.stats.var.append(np.var(out))
        i+=1
    results[th] = layers
