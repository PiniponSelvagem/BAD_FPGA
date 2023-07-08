import os

import time
from datetime import timedelta

from microfaune.detection import RNNDetector
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



startModel = time.time()
#########################################################



detector = RNNDetector()
model = detector.model

#model.save('model/model.h5')

#########################################################
end = time.time()
elapsedModel = end - startModel





startConvert = time.time()
#########################################################

directory = "tflite_models\\"
saveFileName = "microfaune_model_"



import numpy as np
audioDir = "..\\..\\..\\files_audio\\ff1010bird\\wav\\"
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


"""
#https://stackoverflow.com/questions/57877959/what-is-the-correct-way-to-create-representative-dataset-for-tfliteconverter
BATCH_SIZE = 200
NORM_H = 1024
NORM_W = 1024
files = os.listdir(audioDir)
def rep_data_gen():
    a = []
    for i in range(BATCH_SIZE):
        file_name = files[i]
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
"""

"""
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]
"""




def rep_data_gen():
    data = np.random.rand(1, 1, 40, 1)
    yield [data.astype(np.float32)]


# https://www.tensorflow.org/lite/performance/model_optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]


converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter=True
converter.representative_dataset = rep_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert and Save the model
tflite_model = converter.convert()


open("model/model_quantized.tflite", "wb").write(tflite_model)



#########################################################
end = time.time()
elapsedConvert = end - startConvert




startEvaluate = time.time()
#########################################################


# prepare test_data
test_class_file = np.loadtxt('./test_data/ff1010bird_metadata.txt', delimiter=',', dtype='str')
test_data = np.load('./test_data/ff1010bird_feature.npy')
wav_names = test_class_file[:,0]

import microfaune.audio as audio

# Taken from microfaune/detection.py
def compute_features(audio_signals):
    X = []
    for data in audio_signals:
        x = audio.create_spec(data, fs=44100, n_mels=40, n_fft=2048,
                        hop_len=1024).transpose()
        X.append(x[..., np.newaxis].astype(np.float32)/255)
    return X

# Taken from microfaune/detection.py"
""" NOT IN USE, JUST EXAMPLE TO SATRT PREDICTING """
def predict_on_wav(wav_file):
    fs, data = audio.load_wav(wav_file)
    X = compute_features([data])
    scores, local_scores = predict(np.array(X))
    return scores[0], local_scores[0]

# Taken from microfaune/detection.py
""" NOT IN USE, JUST EXAMPLE TO SATRT PREDICTING """
def predict(X):
    scores = []
    local_scores = []
    for x in X:
        s, local_s = model.predict(x[np.newaxis, ...])
        scores.append(s[0])
        local_scores.append(local_s.flatten())
    scores = np.array(s)
    return scores, local_scores


def run_tflite_model(tflite_file):
    # Initialize the interpreter
    print("#### Interperter allocating tensors ####")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()
    #
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    #
    print("#### input_details ####")
    print(input_details)
    print("#### output_details ####")
    print(output_details)
    #
    print("#### Predicting... ####")
    global wav_names
    global test_data
    #
    output = []
    #
    i = 0
    total = 10
    #
    input_scale, input_zero_point = input_details["quantization"]
    for data in test_data:
        for x in data:
            x = x / input_scale + input_zero_point
            x = np.rint(x).astype(input_details["dtype"])
            local_scores = []
            for xi in x:
                interpreter.set_tensor(input_details["index"], xi[np.newaxis, np.newaxis, ...])
                interpreter.invoke()
                local_scores.append(interpreter.get_tensor(output_details["index"])[0][0][0])
        #
        score = 0
        for ls in local_scores:
            if ls > score:
                score = ls
        output.append(score)
        #
        #interpreter.set_tensor(input_details["index"], x[0][np.newaxis, np.newaxis, ...])
        #interpreter.invoke()
        #        
        #
        if i%1==0:
            print(str(i) + " / " + str(total))
        i += 1
        if i==10:
            break
    return output


# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type):
    global test_data
    global test_class_file
    #
    predictions = run_tflite_model(tflite_file)
    print(len(predictions))
    #
    #
    # convert predictions to 0 or 1
    predictions_bool = np.zeros((len(test_data),), dtype=str)
    i = 0
    for i in range(len(predictions)):
        if predictions[i] >= 128:
            predictions_bool[i] = 1
        else:
            predictions_bool[i] = 0
    #
    #save_test_labels_&_probs
    #test_class_file=np.array(test_class_file)
    np.savetxt('tflite_result.csv',np.c_[test_class_file[:,0],test_class_file[:,1],predictions_bool[:],predictions[:]],fmt='%s', delimiter=',')
    #
    accuracy = (np.sum(test_class_file[:,1] == predictions_bool) * 100) / len(test_data)
    #
    print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
        model_type, accuracy, len(test_data)))




print("#### Evaluating ####")
evaluate_model("model/model_quantized.tflite", audioDir)

#########################################################
end = time.time()
elapsedEvaluate = end - startEvaluate

print('Time elapsed (Model)   : ' + str(timedelta(seconds=elapsedModel)))
print('Time elapsed (Convert) : ' + str(timedelta(seconds=elapsedConvert)))
print('Time elapsed (Evaluate): ' + str(timedelta(seconds=elapsedEvaluate)))



