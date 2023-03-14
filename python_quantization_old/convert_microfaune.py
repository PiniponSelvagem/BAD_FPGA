import os

from microfaune.detection import RNNDetector
import tensorflow as tf

detector = RNNDetector()
model = detector.model

quantizationMode = 0    # 0->just convert; 1->float16; 2->dynamic; 3->int8
directory = "tflite_models\\"
saveFileName = "microfaune_model_"



import pathlib
from microfaune import audio
import cv2
import numpy as np
audioDir = "..\\..\\files_audio\\ff1010bird\\wav\\"
"""
for file in pathlib.Path(audioDir).iterdir():
    # read and resize the image
    ""
    img = cv2.imread(r"{}".format(file.resolve()))
    new_img = cv2.resize(img, (224, 224))
    ""
    spec = audio.wav2spc(file)

    new_img = cv2.resize(spec, (1, 40))
"""

#https://stackoverflow.com/questions/57877959/what-is-the-correct-way-to-create-representative-dataset-for-tfliteconverter
BATCH_SIZE = 200
NORM_H = 1024
NORM_W = 1024
files = pathlib.Path(audioDir).iterdir()
def rep_data_gen():
    a = []
    for i in range(BATCH_SIZE):
        inst = files[i]
        file_name = inst['filename']
        img = cv2.imread(audioDir + file_name)
        img = cv2.resize(img, (NORM_H, NORM_W))
        img = img / 255.0
        img = img.astype('float32')
        a.append(img)
    a = np.array(a)
    print(a.shape) # a is np array of 160 3D images
    img = tf.data.Dataset.from_tensor_slices(a).batch(1)
    for i in img.take(BATCH_SIZE):
        print(i)
        yield [i]




# https://www.tensorflow.org/lite/performance/model_optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

# Set the optimization mode 
converter.optimizations = [tf.lite.Optimize.DEFAULT]

match quantizationMode:
    case 0: # Convert the model (no quantization)
        saveFileName += "converted"

    case 1: # Convert the model (float16 quantization)
        converter.target_spec.supported_types = [tf.float16]
        saveFileName += "float16"

    case 2: # Convert the model (dynamic quantization)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        saveFileName += "dynamic"
    
    case 3: # Convert the model (int8 quantization)
        """
        currently giving runtime error
        """
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter=True
        converter.representative_dataset = rep_data_gen
        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to uint8
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        saveFileName += "int8"

# Convert and Save the model
tflite_model = converter.convert()
open(directory+saveFileName+".tflite", "wb").write(tflite_model)
