import os

from microfaune.detection import RNNDetector


"""
required:
pip install numpy
pip install tensorflow
pip install librosa
"""

directory = "..\\..\\files_audio\\ff1010bird\\wav\\"
detector = RNNDetector()

model = detector.create_model()

import tensorflow as tf
"""
# Convert the model (no quantization)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

# save the model
open("microfaune_tflite_model.tflite", "wb").write(tflite_model)
"""


# Convert the model (float16 quantization)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

# Set the optimization mode 
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set float16 is the supported type on the target platform
converter.target_spec.supported_types = [tf.float16]

# Convert and Save the model
tflite_model = converter.convert()
open("microfaune_converted_model_float16.tflite", "wb").write(tflite_model)


"""
# Convert the model (dynamic quantization)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the optimization mode 
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

# Set the optimization mode 
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and Save the model
tflite_model = converter.convert()
directory = "tflite_models\\"
open(directory+"microfaune_converted_model_dynamic.tflite", "wb").write(tflite_model)
"""


"""
print("Directory -> "+directory)
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        global_score, local_score = detector.predict_on_wav(f)
        print(filename+" -> "+str(global_score[0]))
    break
"""

"""
import numpy as np
test_class_file = np.loadtxt("test.csv", delimiter=',', dtype='str') # test_class_file[:,0] -> ID, test_class_file[:,1] -> real value of hasBird
result = []
print("Directory -> "+directory)
for filename in test_class_file:
    f = os.path.join(directory, filename+".wav")
    if os.path.isfile(f):
        global_score, local_score = detector.predict_on_wav(f)
        print(filename+" -> "+str(global_score[0]))
        result.append(global_score[0])
        break

result = np.array(result)
np.savetxt("result.csv",np.c_[test_class_file[:],result[:]], fmt='%s', delimiter=',')
#print(len(detector.model.layers))
"""




"""
ficheiro  |      %       | se >50 considerar passaro | há ou não / clareza       | parece volume % (passaro)
----------+--------------+---------------------------+---------------------------+---------------------------
55.wav    | 0.009491051  |            ...            | não tem passaro           |    0 %
100.wav   | 0.25798014   |            ...            | tem passaro, muito subtil |   10 %
377.wav   | 0.75031155   |            SIM            | tem passaro, muito subtil |    5 %
518.wav   | 0.091156006  |            ...            | não tem passaro           |    0 %
1045.wav  | 0.074668966  |            ...            | não tem passaro           |    0 %
1050.wav  | 0.98813665   |            SIM            | tem passaro, muito claro  |   80 %
1051.wav  | 0.9875864    |            SIM            | tem passaro, muito claro  |   95 %
1053.wav  | 0.011163684  |            ...            | não tem passaro           |    0 %
1055.wav  | 0.9821978    |            SIM            | tem passaro, muito claro  |  100 %
2155.wav  | 0.05937126   |            ...            | não tem passaro           |    0 %
2157.wav  | 0.9872018    |            SIM            | tem passaro, muito claro  |  100 %
2432.wav  | 0.012723158  |            ...            | não tem passaro           |    0 %
2521.wav  | 0.037950914  |            ...            | não tem passaro           |    0 %
2527.wav  | 0.017906759  |            ...            | não tem passaro           |    0 %
2536.wav  | 0.9857993    |            SIM            | tem passaro, muito claro  |   98 %
3183.wav  | 0.9569899    |            SIM            | tem passaro, razoável     |   15 %
3191.wav  | 0.80604786   |            SIM            | tem passaro, muito subtil |    5 %
5204.wav  | 0.6984873    |            SIM            | tem passaro, muito subtil |   40 %
5996.wav  | 0.028960453  |            ...            | não tem passaro           |    0 %
8059.wav  | 0.9872508    |            SIM            | tem passaro, muito claro  |  100 %

taxa de sucesso para esta pequena amostra -> 10/11 -> 90%
"""