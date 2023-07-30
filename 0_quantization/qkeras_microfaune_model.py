import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import math

from qkeras.utils import model_quantize

class MicrofauneAI:
    def __init__(self, bits, max_value, padding):
        """
        "mean_quantizer": f"quantized_po2({bits}, {max_value})",    #quantized_relu_po2
        "gamma_quantizer": f"quantized_po2({bits}, {max_value})",   #quantized_relu_po2
        "variance_quantizer": f"quantized_po2({bits}, {max_value})",#quantized_relu_po2
        "beta_quantizer": f"quantized_po2({bits}, {max_value})",    #quantized_relu_po2
        "inverse_quantizer": f"quantized_po2({bits}, {max_value})"  #quantized_relu_po2
        """
        config = {
            "QConv2D": {
                "kernel_quantizer": f"quantized_po2({bits}, {max_value})",
                "bias_quantizer": f"quantized_po2({bits}, {max_value})"
            },
            "QBatchNormalization": {
                "activation": f"quantized_relu_po2({bits}, {max_value})",   # atm havent confirmed if this is necessary
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
        #
        ### model start ###
        n_filter = 64
        conv_reg = keras.regularizers.l2(1e-3)
        #
        spec = keras.Input(shape=[431, 40, 1], dtype=np.float32)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None)(spec)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.MaxPool2D((1, 2))(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.MaxPool2D((1, 2))(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding=padding, kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        if padding == "same":
            x = layers.MaxPool2D((1, 2))(x)
        #
        x = math.reduce_max(x, axis=-2)
        #
        x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
        x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
        #
        x = layers.TimeDistributed(layers.Dense(64, activation="sigmoid"))(x)
        local_pred = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
        pred = math.reduce_max(local_pred, axis=-2)
        ### model end ###
        #
        self.model = keras.Model(inputs=spec, outputs=pred)
        self.dual_model = keras.Model(inputs=spec, outputs=[pred, local_pred]) # for predictions only
        #
        self.model_quant = model_quantize(self.model, config, 4, transfer_weights=True)
        self.dual_model_quant = model_quantize(self.dual_model, config, 4, transfer_weights=True)
        
    def modelOriginal(self):
        return self.model, self.dual_model

    def modelQuantized(self):
        return self.model_quant, self.dual_model_quant
