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

directory = "tflite_models\\"
saveFileName = "microfaune_model_"


pathToTestMetadata = './test_data/ff1010bird_metadata.txt'
pathToTestFeature = './test_data/ff1010bird_feature.npy'

pathToQuantFeature = './feature_extracted/BAD_wblr_feature.npy'



import numpy as np
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


#########################################################
end = time.time()
elapsedModel = end - startModel



import microfaune.audio as audio

def compute_features(audio_signals):
    X = []
    for data in audio_signals:
        x = audio.create_spec(data, fs=44100, n_mels=40, n_fft=2048,
                        hop_len=1024).transpose()
        X.append(x[..., np.newaxis].astype(np.float32)/255)
    return X

def predict_on_wav(wav_file):
    fs, data = audio.load_wav(wav_file)
    X = compute_features([data])
    scores, local_scores = predict(np.array(X))
    return scores[0], local_scores[0]

def predict(X):
    scores = []
    local_scores = []
    for x in X:
        s, local_s = model.predict(x[np.newaxis, ...])
        scores.append(s[0])
        local_scores.append(local_s.flatten())
    scores = np.array(s)
    return scores, local_scores



startConvert = time.time()
#########################################################


audioDir = "..\\..\\..\\files_audio\\warblrb10k_public\\wav\\"
metadata_path = '..\\..\\..\\files_audio_metadata\\warblrb10k_public_metadata.txt'
class_file = np.loadtxt(metadata_path, delimiter=',',dtype='str') # file with metadata in format: ID_OF_WAV,hasBird
wav_names = class_file[:,0]

gen_data = np.load(pathToQuantFeature, allow_pickle=True)
#gen_data -> todas as features de todos os ficheiros wav

def rep_data_gen():
    """
    for data in gen_data:
        datax = np.zeros(shape=(1, 1, 40, 1))
        data = data[0][0]
        i = 0
        for d in data:
            datax[i] = d[0]
        yield [datax.astype(np.float32)]
    """
    for wav_name in wav_names:
        fs, data = audio.load_wav(audioDir+wav_name+".wav")
        X = compute_features([data])
        yield [np.array(X)]

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
test_class_file = np.loadtxt(pathToTestMetadata, delimiter=',', dtype='str')
test_data = np.load(pathToTestFeature)



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
    global test_data
    #
    output = [0] * len(test_data)
    #
    i = 0
    total = len(test_data)
    #
    input_scale, input_zero_point = input_details["quantization"]
    for data in test_data:
        local_scores = []
        for x in data:
            x = x / input_scale + input_zero_point
            x = np.rint(x).astype(input_details["dtype"])
            for xi in x:
                interpreter.set_tensor(input_details["index"], xi[np.newaxis, np.newaxis, ...])
                interpreter.invoke()
                local_scores.append(interpreter.get_tensor(output_details["index"])[0][0][0])
                break
        #
        score = 0
        for ls in local_scores:
            if ls > score:
                score = ls
        output.append(score)
        #
        if i%100==0:
            print(str(i) + " / " + str(total))
        i += 1
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
evaluate_model("model/model_quantized.tflite", "Quantized")


#########################################################
end = time.time()
elapsedEvaluate = end - startEvaluate

print('Time elapsed (Model)   : ' + str(timedelta(seconds=elapsedModel)))
print('Time elapsed (Convert) : ' + str(timedelta(seconds=elapsedConvert)))
print('Time elapsed (Evaluate): ' + str(timedelta(seconds=elapsedEvaluate)))


i = 0
if i==0:
    exit()













startEvaluate = time.time()
############################################


test_class_file = np.loadtxt(pathToTestMetadata, delimiter=',', dtype='str')
test_data = np.load(pathToTestFeature)

tflite_file = "model/model_quantized.tflite"
interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("#### input_details ####")
print(input_details)
print("#### output_details ####")
print(output_details)

print("#### Predicting... ####")

output = []

i = 0
total = len(test_data)

input_scale, input_zero_point = input_details["quantization"]
for data in test_data:
    local_scores = []
    for x in data:
        x = x / input_scale + input_zero_point
        x = np.rint(x).astype(input_details["dtype"])
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
    if i%1==0:
        print(str(i) + " / " + str(total) + " -> " + str(output[i]))
    i += 1
    if i==10:
        break

output



#########################################################
end = time.time()
elapsedEvaluate = end - startEvaluate

print('Time elapsed (Evaluate): ' + str(timedelta(seconds=elapsedEvaluate)))

















#################################################





import microfaune.audio as audio

def compute_features(audio_signals):
    X = []
    for data in audio_signals:
        x = audio.create_spec(data, fs=44100, n_mels=40, n_fft=2048,
                        hop_len=1024).transpose()
        X.append(x[..., np.newaxis].astype(np.float32)/255)
    return X


###############

fs, data = audio.load_wav("37798.wav")
X = compute_features([data])

scores = []
local_scores = []
for x in X:
    s, local_s = model.predict(x[np.newaxis, ...])
    scores.append(s[0])
    local_scores.append(local_s.flatten())

scores = np.array(s)
scores






fs, data = audio.load_wav("37798.wav")
X = compute_features([data])
X = np.multiply(X, 2**8)
X = X.round(decimals=0)
X = np.divide(X, 2**8)

scores = []
local_scores = []
for x in X:
    s, local_s = model.predict(x[np.newaxis, ...])
    scores.append(s[0])
    local_scores.append(local_s.flatten())

scores = np.array(s)
scores



###############






fs, data = audio.load_wav("37798.wav")
X = compute_features([data])

output = []

input_scale, input_zero_point = input_details["quantization"]
#for data in test_data:
scores = []
local_scores = []
for x in X:
    # passar de float para inteiros
    x = x / input_scale + input_zero_point
    x = np.rint(x).astype(input_details["dtype"])
    #
    #s, local_s = model.predict(x[np.newaxis, ...])
    for xi in x:
        interpreter.set_tensor(input_details["index"], xi[np.newaxis, np.newaxis, ...])
        interpreter.invoke()
        #
        #scores.append(s[0])
        #local_scores.append(local_s.flatten())
        local_scores.append(interpreter.get_tensor(output_details["index"])[0][0][0])
    #
    score = 0
    for ls in local_scores:
        if ls > score:
            score = ls
    #
    scores.append(score)



# passar de float para inteiros
    x = x / input_scale + input_zero_point
    x = np.rint(x).astype(input_details["dtype"])
    #
    #s, local_s = model.predict(x[np.newaxis, ...])
    interpreter.set_tensor(input_details["index"], x[np.newaxis, ...])
    interpreter.invoke()
    #
    #scores.append(s[0])
    #local_scores.append(local_s.flatten())
    scores.append(interpreter.get_tensor(output_details["index"]))


score = 0
for ls in local_scores:
    if ls > score:
        score = ls

output.append(score)



###############



tflite_file = "model/model_quantized.tflite"
interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details["quantization"]

fs, data = audio.load_wav("37798.wav")
X = compute_features([data])
X = np.array(X)

scores = []
local_scores = []
for x in X:
    # passar de float para inteiros
    x = x / input_scale + input_zero_point
    x = np.rint(x).astype(input_details["dtype"])
    #
    #s, local_s = model.predict(x[np.newaxis, ...])
    interpreter.set_tensor(input_details["index"], x[np.newaxis, ...])
    interpreter.invoke()
    #
    #scores.append(s[0])
    #local_scores.append(local_s.flatten())
    scores.append(interpreter.get_tensor(output_details["index"]))























#s, local_s = model.predict(x[np.newaxis, ...])
interpreter.set_tensor(input_details["index"], x[np.newaxis, ...])
interpreter.invoke()
#scores.append(s[0])
#local_scores.append(local_s.flatten())
interpreter.get_tensor(output_details["index"])








output = [0] * len(test_data)

i = 0
total = len(test_data)

input_scale, input_zero_point = input_details["quantization"]
for data in test_data:
    local_scores = []
    for x in data:
        x = x / input_scale + input_zero_point
        x = np.rint(x).astype(input_details["dtype"])
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
    if i%100==0:
        print(str(i) + " / " + str(total))
    i += 1
    break

