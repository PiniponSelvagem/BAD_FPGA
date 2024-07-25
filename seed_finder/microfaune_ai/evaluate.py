from microfaune.detection import RNNDetector

detector = RNNDetector()

import numpy as np
pathToTestWavs = '../../files_audio/birdvox20k/wav/'
pathToTestMetadata = '../../files_audio_metadata/birdvox20k_metadata.txt'
test_class_file = np.loadtxt(pathToTestMetadata, delimiter=',', dtype='str') # test_class_file[:,0] -> ID, test_class_file[:,1] -> real value of hasBird


total = len(test_class_file)
scores = []
for i in range(len(test_class_file)):
    global_score, local_score = detector.predict_on_wav(pathToTestWavs+test_class_file[i][0]+'.wav')
    scores.append(global_score[0])

nCorrect = 0
real_values = test_class_file[:,1]
scores_bin = []
for i in range(len(scores)):
    if scores[i] >= 0.5:
        scores_bin.append(1)
    else:
        scores_bin.append(0)

for i in range(total):
    nCorrect = nCorrect + (int(real_values[i]) == scores_bin[i])

accuracy = nCorrect/total
print("Accuracy for '"+pathToTestWavs+"' dataset: "+str(accuracy))

