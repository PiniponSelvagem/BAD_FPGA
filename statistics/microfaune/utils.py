import numpy as np
import tensorflow as tf

class Layer:
    def __init__(self, name, shape, stats):
        self.name = name
        self.shape = shape
        self.stats = stats
    def encode(self):
        return self.__dict__
    def __str__(self):
        return f"Layer(name={self.name}, shape={self.shape}, stats={str(self.stats)})"


class Stats:
    def __init__(self, min, max, range, mean, median, std, var):
        self.min = min
        self.max = max
        self.range = range
        self.mean = mean
        self.median = median
        self.std = std
        self.var = var
    def encode(self):
        return self.__dict__
    def __str__(self):
        return f"Stats(min={self.min}, max={self.max}, range={self.range}, mean={self.mean}, median={self.median}, std={self.std}, var={self.var})"


def getOutputOfLayer(layer_name, model, feature):
    features = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(name=layer_name).output,
    )
    return features(np.array(feature))



"""
import multiprocessing

def calculate_statistics(i, model, group, results):
    # do some calculation
    pass

if __name__ == '__main__':
    num_processes = 4
    pool = multiprocessing.Pool(processes=num_processes)
    results = []
    for i in range(num_processes):
        result = pool.apply_async(calculate_statistics, args=(i, model, group, results))
        results.append(result)
    pool.close()
    pool.join()
    for result in results:
        result = result.get()
    # do something with the results
"""