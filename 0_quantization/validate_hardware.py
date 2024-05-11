import os
import time
from datetime import timedelta

import numpy as np
import csv
import random
random.seed(666)

from microfaune.detection import RNNDetector
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import qkeras_microfaune_model_1cell as qmodel
from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1 import ModelConfig

import cutils
from bitarray import bitarray


dataset = "ff1010bird"
nSelByClass = 8       # Will select X with 0 class and X with 1 class, total will the X*2 and the 1st half is all 0 class and 2nd half all 1 class.
basePath = "/mnt/e/Rodrigo/ISEL/2_Mestrado/2-ANO_1-sem/TFM"

directory = "validate_hardware/"
binName = "input_"+str(nSelByClass*2)





# Load original model
detector = RNNDetector()
modelOrig = detector.model

# Load quantized model
modelQuant_folder = ModelConfig.folder
modelQuant_name = ModelConfig.name
_, modelQuant = qmodel.MicrofauneAI(ModelConfig).modelQuantized()
modelQuant.load_weights(f"{modelQuant_folder}/{modelQuant_name}.h5")



#### Taken and simplified from dump io script ####
data_type = {}
data_type["name"] = "ap_fixed"
data_type["bits_total"] = 4
data_type["bits_int"] = 0

doPaddingInput = True
START_PADDING = 64   # number of padding to place on the z axis with zeros to achieve padding for HLS

folderDumpIO = "validate_hardware"

if not os.path.isdir(folderDumpIO):
    os.makedirs(folderDumpIO)
##################################################


startModel = time.time()
#########################################################

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




class Stats:
    def __init__(self, name, original, quantized):
        self.name = name
        self.original = original
        self.quantized = quantized
    def setQuantized(self, result):
        self.quantized = result
    def toList(self):
        return [self.name, self.original, self.quantized]

def saveStatsToCSV(stats, csvName):
    with open(directory+csvName+'.csv', 'w', newline='') as csvfile:
        fieldnames = ['Name', 'Original', 'Quantized']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats:
            writer.writerow(dict(zip(fieldnames, stat.toList())))

def getInputBin(fileName, X):
    """
    Simplified version of qKeras_microfaune_dump_io.py that only saves the q_activation, aka quantized input, to bin file
    """
    layers_names = []
    for layer in modelQuant.layers:
        layers_names.append(layer.name)
    #
    def getOutputOfLayer(layer_name, X):
        features = tf.keras.models.Model(
            inputs=modelQuant.inputs,
            outputs=modelQuant.get_layer(name=layer_name).output,
        )
        return features(np.array(X))
    #
    out = getOutputOfLayer("q_activation", X)
    if doPaddingInput:  # if layer_name == "q_activation":
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
    #
    return cutils.saveArray_bin(folderDumpIO, fileName, np.array(out), data_type, saveBinAsInteger=False, binPositiveOnly=True, nPacket=1, saveFile=False)

def saveBinPack(file_path, packed_bits):
    packed_data = b''
    while len(packed_bits) % 8 != 0:
        # Pad the packed bits to a multiple of 8 if necessary
        packed_bits.append(False)
    #
    packed_data = packed_bits.tobytes()
    with open(file_path, 'wb') as file:
        file.write(packed_data)

def predictAll(selected_files, modelOrig, modelQuant):
    stats = np.array([])
    binData = bitarray()
    for file in selected_files:
        fs, data = load_wav(audioDir+file+".wav")
        X = compute_features([data])
        scoreOrig, _  = modelOrig.predict(np.array(X))
        scoreQuant, _ = modelQuant.predict(np.array(X))
        packet_bits, _ = getInputBin(file, X)
        binData.extend(packet_bits)
        stats = np.append(stats, Stats(file, scoreOrig[0][0], scoreQuant[0][0]))
    return stats, binData


start = time.time()
#########################################################


audioDir = basePath+"/files_audio/"+dataset+"/wav/"
metadata_path = basePath+"/files_audio_metadata/"+dataset+"_metadata.txt"
class_file = np.loadtxt(metadata_path, delimiter=',',dtype='str') # file with metadata in format: ID_OF_WAV,hasBird

# Separate WAVs by their classification
files_0 = [obj[0] for obj in class_file if obj[1] == "0"]
files_1 = [obj[0] for obj in class_file if obj[1] == "1"]

# Shuffle and select
random.shuffle(files_0)
random.shuffle(files_1)
selected_files_0 = files_0[:nSelByClass]
selected_files_1 = files_1[:nSelByClass]
print("selected_files_0 size is '"+str(len(selected_files_0))+"'")
print("selected_files_1 size is '"+str(len(selected_files_1))+"'")
assert len(selected_files_0) == nSelByClass
assert len(selected_files_1) == nSelByClass

# Predict selection
stats_0, bin_data_0 = predictAll(selected_files_0, modelOrig, modelQuant)
stats_1, bin_data_1 = predictAll(selected_files_1, modelOrig, modelQuant)

bin_data_0.extend(bin_data_1)
binData = bin_data_0
saveBinPack(directory+binName+".bin", binData)

# Save stats to CSV
saveStatsToCSV(stats_0, "stats_0")
saveStatsToCSV(stats_1, "stats_1")












#########################################################
end = time.time()
elapsed = end - start


print('Time elapsed: ' + str(timedelta(seconds=elapsed)))

