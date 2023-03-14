import os
import numpy as np
import microfaune.audio as audio

"""
audioDir = "..\\..\\..\\files_audio\\ff1010bird\\wav\\"

class_file = np.loadtxt('./test_data/ff1010bird_metadata.txt', delimiter=',', dtype='str')
wav_names = class_file[:,0]

saveFeatureTo = './test_data/' # path where to save the extrated features
baseName = 'ff1010bird' # base file name
"""


audioDir = "..\\..\\..\\files_audio\\warblrb10k_public\\wav\\"
metadata_path = '..\\..\\..\\files_audio_metadata\\warblrb10k_public_metadata.txt'
class_file = np.loadtxt(metadata_path, delimiter=',',dtype='str') # file with metadata in format: ID_OF_WAV,hasBird
wav_names = class_file[:,0]

saveFeatureTo = './feature_extracted/' # path where to save the extrated features
baseName = 'BAD_wblr' # base file name for feature and label





#class_name_&_class_label
class_names = class_file[:,0]
class_labels = class_file[:,1]

#array_to_store_feature_&_label
feature_file = []
label_file = []

# Taken from microfaune/detection.py
def compute_features(audio_signals):
    X = []
    for data in audio_signals:
        x = audio.create_spec(data, fs=44100, n_mels=40, n_fft=2048,
                        hop_len=1024).transpose()
        X.append(x[..., np.newaxis].astype(np.float32)/255)
    return X


def create_features(wav_file):
    fs, data = audio.load_wav(wav_file)
    X = compute_features([data])
    return X


for wav_filename in wav_names:
    X = create_features(audioDir+wav_filename+".wav")
    feature_file.append(X)



if not os.path.exists(saveFeatureTo):
    os.mkdir(saveFeatureTo)

print(len(feature_file))
np.save(saveFeatureTo+baseName+'_feature',feature_file)