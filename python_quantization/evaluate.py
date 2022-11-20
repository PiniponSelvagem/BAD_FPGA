import os
import numpy as np
from microfaune_ai.microfaune.detection import RNNDetector

"""
required:
pip install numpy
pip install tensorflow
pip install librosa
"""

import time
from datetime import timedelta

start = time.time()


directory = "..\\..\\files_audio\\warblrb10k_evaluate\\wav\\"
detector = RNNDetector()

class_file = np.loadtxt('run-B.csv', delimiter=',', dtype='str')
wav_names = class_file[:,0]


with open("result.csv", "w") as resultFile:
    for wav_name in wav_names:
        filename = wav_name+'.wav'
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            global_score, local_score = detector.predict_on_wav(f)  # global_score is the max found in local_score[]
            print(filename+" -> "+str(global_score[0]))
            resultFile.write(filename+','+str(global_score[0])+'\n')


end = time.time()
elapsed = end - start

print('Time elapsed: ' + str(timedelta(seconds=elapsed)))
