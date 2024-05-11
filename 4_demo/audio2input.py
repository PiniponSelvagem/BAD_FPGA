import numpy as np
from scipy.io import wavfile
from scipy import signal
import librosa

import tensorflow as tf
import random
random.seed(666)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import cutils
import model_quantized.qkeras_microfaune_model_1cell as qmodel
from model_quantized.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1 import ModelConfig

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



#### Taken from microfaune.audio ####
def load_wav(path, decimate=None):
    """Load audio data.

        Parameters
        ----------
        path: str
            Wav file path.
        decimate: int
            If not None, downsampling by a factor of `decimate` value.

        Returns
        -------
        S: array-like
            Array of shape (Mel bands, time) containing the spectrogram.
    """
    fs, data = wavfile.read(path)
    data = data.astype(np.float32)
    if decimate is not None:
        data = signal.decimate(data, decimate)
        fs /= decimate
    return fs, data

#### Taken from microfaune.audio ####
def create_spec(data, fs, n_mels=32, n_fft=2048, hop_len=1024):
    """Compute the Mel spectrogram from audio data.

        Parameters
        ----------
        data: array-like
            Audio data.
        fs: int
            Sampling frequency in Hz.
        n_mels: int
            Number of Mel bands to generate.
        n_fft: int
            Length of the FFT window.
        hop_len: int
            Number of samples between successive frames.

        Returns
        -------
        S: array-like
            Array of shape (Mel bands, time) containing the spectrogram.
    """
    # Calculate spectrogram
    S = librosa.feature.melspectrogram(
    y=data, sr=fs, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
    S = S.astype(np.float32)
    # Convert power to dB
    S = librosa.power_to_db(S)
    return S

#### Taken from microfaune.detection ####
def compute_features(audio_signals):
    """Compute features on audio signals.

    Parameters
    ----------
    audio_signals: list
        Audio signals of possibly various lengths.

    Returns
    -------
    X: list
        Features for each audio signal
    """
    X = []
    for data in audio_signals:
        x = create_spec(data, fs=44100, n_mels=40, n_fft=2048,
                        hop_len=1024).transpose()
        X.append(x[..., np.newaxis].astype(np.float32)/255)
    return X


def getInputBin(X):
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
    return cutils.saveArray_bin("", "", np.array(out), data_type, saveBinAsInteger=False, binPositiveOnly=True, nPacket=1, saveFile=False)

def saveBinPack(file_path, packed_bits):
    packed_data = b''
    while len(packed_bits) % 8 != 0:
        # Pad the packed bits to a multiple of 8 if necessary
        packed_bits.append(False)
    #
    packed_data = packed_bits.tobytes()
    with open(file_path, 'wb') as file:
        file.write(packed_data)

def getHardwareInput(audioPath):
    _, data = load_wav(audioPath)
    X = compute_features([data])
    packet_bits, _ = getInputBin(X)
    return packet_bits