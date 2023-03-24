SEED = 43

import os
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import random

import numpy as np
from microfaune.detection import RNNDetector

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ['PYTHONHASHSEED']=str(SEED)
tf.compat.v1.reset_default_graph()
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import math
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam

import qkeras
from qkeras import *
from qkeras import QMaxPooling2D, QReduceMax, QModel
from qkeras.utils import model_save_quantized_weights

NB_EPOCH = 100
BATCH_SIZE = 64
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam(lr=0.0001, decay=0.000025)
VALIDATION_SPLIT = 0.1


pathToTrainFeature = './feature_extracted/BAD_wblr_feature.npy'
pathToTrainLabel = './feature_extracted/BAD_wblr_label.npy'

saveModelTo = './model/' # path where to save generated model and csv file with results

pathToTestMetadata = './tester/ff1010bird_metadata.txt'
pathToTestFeature = './tester/ff1010bird_feature.npy'


# create model
n_filter = 64

spec = x_in = layers.Input(shape=[None, 40, 1], dtype=np.float32)
x = QConv2D(n_filter, (3, 3), padding="same", kernel_quantizer="quantized_bits(8,0)", bias_quantizer="quantized_bits(8,0)")(spec)
x = QBatchNormalization()(x)
x = QActivation("quantized_relu(8,0)")(x)
x = QConv2D(n_filter, (3, 3), padding="same", kernel_quantizer="quantized_bits(8,0)", bias_quantizer="quantized_bits(8,0)")(x)
x = QBatchNormalization()(x)
x = QActivation("quantized_relu(8,0)")(x)
x = layers.MaxPooling2D(pool_size=(1, 2))(x)

x = QConv2D(n_filter, (3, 3), padding="same", kernel_quantizer="quantized_bits(8,0)", bias_quantizer="quantized_bits(8,0)")(x)
x = QBatchNormalization()(x)
x = QActivation("quantized_relu(8,0)")(x)
x = QConv2D(n_filter, (3, 3), padding="same", kernel_quantizer="quantized_bits(8,0)", bias_quantizer="quantized_bits(8,0)")(x)
x = QBatchNormalization()(x)
x = QActivation("quantized_relu(8,0)")(x)
x = layers.MaxPooling2D(pool_size=(1, 2))(x)

x = QConv2D(n_filter, (3, 3), padding="same", kernel_quantizer="quantized_bits(8,0)", bias_quantizer="quantized_bits(8,0)")(x)
x = QBatchNormalization()(x)
x = QActivation("quantized_relu(8,0)")(x)
x = QConv2D(n_filter, (3, 3), padding="same", kernel_quantizer="quantized_bits(8,0)", bias_quantizer="quantized_bits(8,0)")(x)
x = QBatchNormalization()(x)
x = QActivation("quantized_relu(8,0)")(x)
x = layers.MaxPooling2D(pool_size=(1, 2))(x)

x = math.reduce_max(x, axis=-2)

x = QBidirectional(QGRU(64, return_sequences=True))(x)
x = QBidirectional(QGRU(64, return_sequences=True))(x)

x = TimeDistributed(QDense(64, activation="sigmoid", bias_initializer="zeros", 
                            kernel_quantizer="quantized_bits(4,0,1)", bias_quantizer="quantized_bits(4,0,1)"))(x)
local_pred = TimeDistributed(QDense(1, activation="sigmoid", bias_initializer="zeros", 
                                    kernel_quantizer="quantized_bits(4,0,1)", bias_quantizer="quantized_bits(4,0,1)"))(x)
pred = QReduceMax(axis=-2)(local_pred)
model = QModel(inputs=spec, outputs=[pred, local_pred])





from sklearn.model_selection import train_test_split

# train_data
classes = 2
feature = np.load(pathToTrainFeature)
label = np.load(pathToTrainLabel)
label = to_categorical(label, 2)
opt = Adam(decay = 1e-6)
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, shuffle=True)





# train model
history = model.fit(
    x_train, y_train, batch_size=BATCH_SIZE,
    epochs=NB_EPOCH, initial_epoch=1, verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT)

outputs = []
output_names = []

for layer in model.layers:
    if layer.__class__.__name__ in ["QActivation", "Activation",
                                "QDense", "QConv2D", "QDepthwiseConv2D"]:
        output_names.append(layer.name)
        outputs.append(layer.output)

model_debug = Model(inputs=[x_in], outputs=outputs)

outputs = model_debug.predict(x_train)

print("{:30} {: 8.4f} {: 8.4f}".format(
    "input", np.min(x_train), np.max(x_train)))

for n, p in zip(output_names, outputs):
    print("{:30} {: 8.4f} {: 8.4f}".format(n, np.min(p), np.max(p)), end="")
    layer = model.get_layer(n)
    for i, weights in enumerate(layer.get_weights()):
        weights = K.eval(layer.get_quantizers()[i](K.constant(weights)))
        print(" ({: 8.4f} {: 8.4f})".format(np.min(weights), np.max(weights)),
            end="")
        print("")

p_test = model.predict(x_test)
p_test.tofile("p_test.bin")

score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])

all_weights = []
model_save_quantized_weights(model)

for layer in model.layers:
    for w, weights in enumerate(layer.get_weights()):
        print(layer.name, w)
        all_weights.append(weights.flatten())

all_weights = np.concatenate(all_weights).astype(np.float32)
print(all_weights.size)





for layer in model.layers:
  for w, weight in enumerate(layer.get_weights()):
    print(layer.name, w, weight.shape)

print_qstats(model)