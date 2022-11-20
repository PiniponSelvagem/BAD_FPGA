#https://heartbeat.comet.ml/running-tensorflow-lite-image-classification-models-in-python-92ef44b4cd47


import numpy as np
import tensorflow as tf
import pathlib

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model_float16.tflite")

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

# input details
print(input_details)
# output details
print(output_details)




import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

def graph_spectrogram(wav_file):
    rate, data = wavfile.read(wav_file)
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate)
    ax.axis('off')
    fig.savefig('spec.png', dpi=300)





import audio
import cv2
directory = "..\\..\\..\\files_audio\\ff1010bird\\wav\\"
for file in pathlib.Path(directory).iterdir():
    
    # read and resize the image
    """
    img = cv2.imread(r"{}".format(file.resolve()))
    new_img = cv2.resize(img, (224, 224))
    """
    spec = audio.wav2spc(file)

    new_img = cv2.resize(spec, (1, 40))

    # input_details[0]['index'] = the index which accepts the input
    interpreter.set_tensor(input_details[0]['index'], [new_img])
    
    # run the inference
    interpreter.invoke()
    
    # output_details[0]['index'] = the index which provides the input
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print("For file {}, the output is {}".format(file.stem, output_data))