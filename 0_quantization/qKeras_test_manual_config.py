import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import math

from qkeras.utils import model_quantize
from qkeras import *


bits = "2"
max_value = "1"
padding = "same"

### model start ###
n_filter = 64
conv_reg = keras.regularizers.l2(1e-3)

spec = keras.Input(shape=[None, 40, 1], dtype=np.float32)

x = QConv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None, kernel_quantizer=f"quantized_po2({bits}, {max_value})", bias_quantizer=f"quantized_po2({bits}, {max_value})")(spec)
x = QBatchNormalization(momentum=0.95, mean_quantizer=f"quantized_relu_po2({bits}, {max_value})", gamma_quantizer=f"quantized_relu_po2({bits}, {max_value})", variance_quantizer=f"quantized_relu_po2({bits}, {max_value})", beta_quantizer=f"quantized_relu_po2({bits}, {max_value})")(x)
x = layers.ReLU()(x)

x = QConv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None, kernel_quantizer=f"quantized_po2({bits}, {max_value})", bias_quantizer=f"quantized_po2({bits}, {max_value})")(spec)
x = QBatchNormalization(momentum=0.95, mean_quantizer=f"quantized_relu_po2({bits}, {max_value})", gamma_quantizer=f"quantized_relu_po2({bits}, {max_value})", variance_quantizer=f"quantized_relu_po2({bits}, {max_value})", beta_quantizer=f"quantized_relu_po2({bits}, {max_value})")(x)
x = layers.ReLU()(x)

x = layers.MaxPool2D((1, 2))(x)

x = QConv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None, kernel_quantizer=f"quantized_po2({bits}, {max_value})", bias_quantizer=f"quantized_po2({bits}, {max_value})")(spec)
x = QBatchNormalization(momentum=0.95, mean_quantizer=f"quantized_relu_po2({bits}, {max_value})", gamma_quantizer=f"quantized_relu_po2({bits}, {max_value})", variance_quantizer=f"quantized_relu_po2({bits}, {max_value})", beta_quantizer=f"quantized_relu_po2({bits}, {max_value})")(x)
x = layers.ReLU()(x)

x = QConv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None, kernel_quantizer=f"quantized_po2({bits}, {max_value})", bias_quantizer=f"quantized_po2({bits}, {max_value})")(spec)
x = QBatchNormalization(momentum=0.95, mean_quantizer=f"quantized_relu_po2({bits}, {max_value})", gamma_quantizer=f"quantized_relu_po2({bits}, {max_value})", variance_quantizer=f"quantized_relu_po2({bits}, {max_value})", beta_quantizer=f"quantized_relu_po2({bits}, {max_value})")(x)
x = layers.ReLU()(x)

x = layers.MaxPool2D((1, 2))(x)

x = QConv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None, kernel_quantizer=f"quantized_po2({bits}, {max_value})", bias_quantizer=f"quantized_po2({bits}, {max_value})")(spec)
x = QBatchNormalization(momentum=0.95, mean_quantizer=f"quantized_relu_po2({bits}, {max_value})", gamma_quantizer=f"quantized_relu_po2({bits}, {max_value})", variance_quantizer=f"quantized_relu_po2({bits}, {max_value})", beta_quantizer=f"quantized_relu_po2({bits}, {max_value})")(x)
x = layers.ReLU()(x)

x = QConv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None, kernel_quantizer=f"quantized_po2({bits}, {max_value})", bias_quantizer=f"quantized_po2({bits}, {max_value})")(spec)
x = QBatchNormalization(momentum=0.95, mean_quantizer=f"quantized_relu_po2({bits}, {max_value})", gamma_quantizer=f"quantized_relu_po2({bits}, {max_value})", variance_quantizer=f"quantized_relu_po2({bits}, {max_value})", beta_quantizer=f"quantized_relu_po2({bits}, {max_value})")(x)
x = layers.ReLU()(x)

x = layers.MaxPool2D((1, 2))(x)    

x = math.reduce_max(x, axis=-2)

x = QBidirectional(QGRU(64, return_sequences=True))(x)
x = QBidirectional(QGRU(64, return_sequences=True))(x)

x = layers.TimeDistributed(QDense(64, activation="sigmoid", kernel_quantizer=f"quantized_po2({bits}, {max_value})", bias_quantizer=f"quantized_po2({bits}, {max_value})"))(x)
local_pred = layers.TimeDistributed(QDense(1, activation="sigmoid", kernel_quantizer=f"quantized_po2({bits}, {max_value})", bias_quantizer=f"quantized_po2({bits}, {max_value})"))(x)
pred = math.reduce_max(local_pred, axis=-2)
### model end ###

model = keras.Model(inputs=spec, outputs=pred)
qmodel = model_quantize(model, None, activation_bits=4, transfer_weights=True)


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


