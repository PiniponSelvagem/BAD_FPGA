from tensorflow import keras
from qkeras.utils import model_quantize


class ModelConfig:
    bits = 4
    integer = 1
    integer_gru_recact = 0
    symmetric = 1
    padding = "same"
    #
    ################
    # Custom Model #
    ################
    def customModel():
        bits = ModelConfig.bits
        integer = ModelConfig.integer
        integer_gru_recact = ModelConfig.integer_gru_recact
        symmetric = ModelConfig.symmetric
        #
        import numpy as np
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow import math
        from qkeras import QConv2DBatchnorm, QActivation, QBidirectional, QGRU
        ### model start ###
        n_filter = 2
        conv_reg = keras.regularizers.l2(1e-3)
        #
        spec = keras.Input(shape=[4, 3, 1], dtype=np.float32)
        #
        x = QConv2DBatchnorm(
            n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None,
            kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            momentum=0.95
        )(spec)
        x = QActivation(f"quantized_relu({bits},{integer})")(x)
        #
        x = QConv2DBatchnorm(
            n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None,
            kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            momentum=0.95
        )(x)
        x = QActivation(f"quantized_relu({bits},{integer})")(x)
        #
        x = layers.MaxPool2D((1, 2))(x)
        #
        x = math.reduce_max(x, axis=-2)
        #
        x = QBidirectional(
            QGRU(n_filter,
                activation = f"quantized_tanh({bits})",
                recurrent_activation = f"quantized_relu({bits},{integer_gru_recact},{symmetric})",
                kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
                recurrent_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
                bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
                #state_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
                return_sequences = True
            )
        )(x)
        x = QBidirectional(
            QGRU(n_filter,
                activation = f"quantized_tanh({bits})",
                recurrent_activation = f"quantized_relu({bits},{integer_gru_recact},{symmetric})",
                kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
                recurrent_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
                bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
                #state_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
                return_sequences = True
            )
        )(x)
        #
        x = layers.TimeDistributed(layers.Dense(n_filter, activation="sigmoid"))(x)
        local_pred = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
        pred = math.reduce_max(local_pred, axis=-2)
        #
        #
        return [spec, pred, local_pred]

class SmallModelMicrofauneAI:
    def __init__(self):
        spec, pred, local_pred = ModelConfig.customModel()
        #
        self.model = keras.Model(inputs=spec, outputs=pred)
        self.dual_model = keras.Model(inputs=spec, outputs=[pred, local_pred]) # for predictions only
        #
        self.model_quant = model_quantize(self.model, None, ModelConfig.bits, transfer_weights=True)
        self.dual_model_quant = model_quantize(self.dual_model, None, ModelConfig.bits, transfer_weights=True)
    #    
    def modelOriginal(self):
        return self.model, self.dual_model
    #
    def modelQuantized(self):
        return self.model_quant, self.dual_model_quant
