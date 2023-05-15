import os
import time
from datetime import timedelta

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import json

from utils import Stats
from calcstats import calculate_statistics

import multiprocessing as mp

# Extend the JSONEncoder class
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, Stats):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)


num_threads = 32
audiofolder = "../../../files_audio/ff1010bird/wav/"
statsfolder = "statistics"
statsfile = "layers.json"

wav_files = []
for file in os.listdir(audiofolder):
    if file.endswith('.wav'):
        wav_files.append(file)

if not os.path.isdir(statsfolder):
    os.makedirs(statsfolder)



start = time.time()
#########################################################

wav_files = []
for file in os.listdir(audiofolder):
    if file.endswith('.wav'):
        wav_files.append(file)

# determine the number of threads to use
num_arrays = len(wav_files)

# determine the size of each group
group_size = (num_arrays + num_threads - 1) // num_threads
wav_files_groups = [wav_files[i:i+group_size] for i in range(0, len(wav_files), group_size)]

# create a list to store the thread objects
threads = []

# create a list to store the results
results = []

res = []
#if __name__ == '__main__':

# create processes
processes = []
i = 0
if __name__ == '__main__':
    manager = mp.Manager()
    results = manager.dict()
    for group in wav_files_groups:
        p = mp.Process(target=calculate_statistics, args=(i, audiofolder, group, results))
        processes.append(p)
        i+=1

    # start processes
    for p in processes:
        p.start()

    # wait for processes to finish
    for p in processes:
        p.join()

    layers = []
    for th in results:
        for layer in results[th]:
            layer_exists = False
            for le in layers:
                if le.name == layer.name:
                    # combine the statistics of the two layers
                    le.stats.min = np.min([np.min(le.stats.min), np.min(layer.stats.min)])
                    le.stats.max = np.max([np.max(le.stats.max), np.max(layer.stats.max)])
                    le.stats.range = le.stats.max - le.stats.min
                    le.stats.mean = le.stats.mean + layer.stats.mean
                    le.stats.median = le.stats.median + layer.stats.median
                    le.stats.std = le.stats.std + layer.stats.std
                    le.stats.var = le.stats.var + layer.stats.var
                    layer_exists = True
                    break
            if not layer_exists:
                layers.append(layer)

    for layer in layers:
        layer.stats.mean = np.mean(layer.stats.mean)
        layer.stats.median = np.mean(layer.stats.median)
        layer.stats.std = np.mean(layer.stats.std)
        layer.stats.var = np.mean(layer.stats.var)

    layers_dict = [layer.__dict__ for layer in layers]
    json_string = json.dumps(layers_dict, cls=NumpyEncoder, indent=4)

    with open(statsfolder+"/"+statsfile, "w") as f:
        f.write(json_string)

    #########################################################
    end = time.time()
    elapsed = end - start


    print('Time elapsed: ' + str(timedelta(seconds=elapsed)))
