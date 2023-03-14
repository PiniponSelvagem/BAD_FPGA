import os
import numpy as np
from microfaune.detection import RNNDetector

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

"""
required:
pip install numpy
pip install tensorflow
pip install librosa
"""

import time
from datetime import timedelta

start = time.time()


directory = "..\\..\\..\\files_audio\\warblrb10k_evaluate\\wav\\"
detector = RNNDetector()

class_file = np.loadtxt('run-B.csv', delimiter=',', dtype='str')
wav_names = class_file[:,0]

"""
with open("result.csv", "w") as resultFile:
    for wav_name in wav_names:
        filename = wav_name+'.wav'
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            global_score, local_score = detector.predict_on_wav(f)  # global_score is the max found in local_score[]
            print(filename+" -> "+str(global_score[0]))
            resultFile.write(filename+','+str(global_score[0])+'\n')
"""




pathToTestMetadata = './test_data/ff1010bird_metadata.txt'
pathToTestFeature = './test_data/ff1010bird_feature.npy'

#test_data
test_class_file = np.loadtxt(pathToTestMetadata, delimiter=',', dtype='str') # test_class_file[:,0] -> ID, test_class_file[:,1] -> real value of hasBird
test_data = np.load(pathToTestFeature)

predict_x = []

#test_label_predict
for wav_name in wav_names:
    filename = wav_name+'.wav'
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        global_score, local_score = detector.predict_on_wav(f)  # global_score is the max found in local_score[]
        #print(filename+" -> "+str(global_score[0]))
        predict_x.append(global_score[0])

for i in range(len(predict_x)):
    pred = predict_x[i]
    if pred > 0.5:
        predict_x[i] = 1
    else:
        predict_x[i] = 0


#save_test_labels_&_probs
pred_probs=np.array(predict_x)
###np.savetxt(saveModelTo+resultName,np.c_[test_class_file[:,0],pred_probs[:]],fmt='%s', delimiter=',')

#show accuracy
total = len(pred_probs)
nCorrect = 0
real_values = test_class_file[:,1]
for i in range(len(pred_probs)):
    nCorrect = nCorrect + (int(real_values[i]) == pred_probs[i])
accuracy = nCorrect/total
print("Accuracy for '"+pathToTestFeature+"' dataset: "+str(accuracy))





end = time.time()
elapsed = end - start

print('Time elapsed: ' + str(timedelta(seconds=elapsed)))
