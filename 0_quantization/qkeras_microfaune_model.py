import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import math

from qkeras.utils import model_quantize

class MicrofauneAI:
    def __init__(self, bits, integer, symmetric, max_value, padding):
        config = {
            "QConv2D": {
                "kernel_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
                "bias_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
            },
            "QBatchNormalization": {
                "mean_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
                "gamma_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
                "variance_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
                "beta_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
                "inverse_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
            },
            "QBidirectional": {
                "activation": f"quantized_tanh({bits})",
                "state_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
                "kernel_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
                "recurrent_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
                "bias_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
            },
        }
        """ # model_weights-quant_bits411
        config = {
            "QConv2D": {
                "kernel_quantizer": "quantized_bits(4,1,1)",
                "bias_quantizer": "quantized_bits(4,1,1)",
            },
            "QBatchNormalization": {
                "mean_quantizer": "quantized_bits(4,1,1)",
                "gamma_quantizer": "quantized_bits(4,1,1)",
                "variance_quantizer": "quantized_bits(4,1,1)",
                "beta_quantizer": "quantized_bits(4,1,1)",
                "inverse_quantizer": "quantized_bits(4,1,1)",
            },
            "QBidirectional": {
                "activation": "quantized_tanh(4)",
                "state_quantizer": "quantized_bits(4,1,1)",
                "kernel_quantizer": "quantized_bits(4,1,1)",
                "recurrent_quantizer": "quantized_bits(4,1,1)",
                "bias_quantizer": "quantized_bits(4,1,1)",
            },
        }
        """
        """ # model_weights-quant_po2_81_conv_gru-quant_bits811_bnorm
        config = {
            "QConv2D": {
                "kernel_quantizer": f"quantized_po2(8,1)",
                "bias_quantizer": f"quantized_po2(8,1)",
            },
            "QBatchNormalization": {
                "mean_quantizer": f"quantized_bits(8,1,1)",
                "gamma_quantizer": f"quantized_bits(8,1,1)",
                "variance_quantizer": f"quantized_bits(8,1,1)",
                "beta_quantizer": f"quantized_bits(8,1,1)",
                "inverse_quantizer": f"quantized_bits(8,1,1)",
            },
            "QBidirectional": {
                "activation": f"quantized_tanh(8)",
                "state_quantizer": f"quantized_po2(8,1)",
                "kernel_quantizer": f"quantized_po2(8,1)",
                "recurrent_quantizer": f"quantized_po2(8,1)",
                "bias_quantizer": f"quantized_po2(8,1)",
            },
        }
        """
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
