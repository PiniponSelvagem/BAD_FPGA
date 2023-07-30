import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import math

from qkeras.utils import model_quantize
from qkeras import *


bits = "2"
max_value = "1"
padding = "same"

config = {
    "QConv2D": {
        "kernel_quantizer": f"quantized_po2({bits}, {max_value})",
        "bias_quantizer": f"quantized_po2({bits}, {max_value})"
    },
    "QBatchNormalization": {
        "mean_quantizer": f"quantized_relu_po2({bits}, {max_value})",
        "gamma_quantizer": f"quantized_relu_po2({bits}, {max_value})",
        "variance_quantizer": f"quantized_relu_po2({bits}, {max_value})",
        "beta_quantizer": f"quantized_relu_po2({bits}, {max_value})",
        "inverse_quantizer": f"quantized_relu_po2({bits}, {max_value})"
        
    },
    "QActivation": {
        "relu": f"quantized_po2({bits}, {max_value})"
    },
    "QGru": {
        "kernel_quantizer": f"quantized_po2({bits}, {max_value})",
        "recurrent_quantizer": f"quantized_po2({bits}, {max_value})",
        "bias_quantizer": f"quantized_po2({bits}, {max_value})"
    },
    "QDense": {
        "kernel_quantizer": f"quantized_po2({bits}, {max_value})",
        "bias_quantizer": f"quantized_po2({bits}, {max_value})"
    }
}

### model start ###
n_filter = 64
conv_reg = keras.regularizers.l2(1e-3)

spec = keras.Input(shape=[431, 40, 1], dtype=np.float32)

x = layers.Conv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None)(spec)
x = layers.BatchNormalization(momentum=0.95)(x)
x = layers.ReLU()(x)

x = layers.Conv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None)(x)
x = layers.BatchNormalization(momentum=0.95)(x)
x = layers.ReLU()(x)

x = layers.MaxPool2D((1, 2))(x)

x = layers.Conv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None)(x)
x = layers.BatchNormalization(momentum=0.95)(x)
x = layers.ReLU()(x)

x = layers.Conv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None)(x)
x = layers.BatchNormalization(momentum=0.95)(x)
x = layers.ReLU()(x)

x = layers.MaxPool2D((1, 2))(x)

x = layers.Conv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None)(x)
x = layers.BatchNormalization(momentum=0.95)(x)
x = layers.ReLU()(x)

x = layers.Conv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None)(x)
x = layers.BatchNormalization(momentum=0.95)(x)
x = layers.ReLU()(x)

x = layers.MaxPool2D((1, 2))(x)

x = math.reduce_max(x, axis=-2)

x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)

x = layers.TimeDistributed(layers.Dense(64, activation="sigmoid"))(x)
local_pred = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
pred = math.reduce_max(local_pred, axis=-2)
### model end ###


model = keras.Model(inputs=spec, outputs=pred)
qmodel = model_quantize(model, config, 4, transfer_weights=True)


import keras.backend as K
for layer in qmodel.layers:
    try:
        if layer.get_quantizers():
            q_gw_w_pairs = [(quantizer, gweight, weight) for quantizer, gweight, weight in zip(layer.get_quantizers(), layer.get_weights(), layer.weights)]
            for _, (quantizer, gweight, weight) in enumerate(q_gw_w_pairs):
                print(weight.name)
                qweight = K.eval(quantizer(gweight))
                print(qweight)
    except AttributeError:
        print("WARNING: weight is not quantized in layer", layer.name)
