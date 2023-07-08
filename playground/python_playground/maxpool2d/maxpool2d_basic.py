import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#########################################################

CUDA_VISIBLE_DEVICES=""     # force use CPU

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


initializer = tf.keras.initializers.Zeros()

spec = layers.Input(shape=[None, 40, 1], dtype=np.float32)
x = layers.MaxPool2D((2, 1), padding="same")(spec)
model = keras.Model(inputs=spec, outputs=[x])

input = np.array([[[ 1.0], [ 2.0], [ 3.0], [ 4.0], [ 5.0], [ 6.0], [ 7.0], [ 8.0], [ 9.0], [10.0],
                   [11.0], [12.0], [13.0], [14.0], [15.0], [16.0], [17.0], [18.0], [19.0], [20.0],
                   [21.0], [22.0], [23.0], [24.0], [25.0], [26.0], [27.0], [28.0], [29.0], [30.0],
                   [31.0], [32.0], [33.0], [34.0], [35.0], [36.0], [37.0], [38.0], [39.0], [40.0]]],
                dtype=np.float32)

predict = model.predict(input)

for j in range(input.shape[0]):
    for i in range(20):
        print(str("{:4}".format(predict[j][i][0][0]))+" ", end='')
    print()

