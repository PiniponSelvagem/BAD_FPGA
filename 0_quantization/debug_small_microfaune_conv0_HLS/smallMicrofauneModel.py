from tensorflow import keras
from qkeras.utils import model_quantize


class ModelConfig:
    bits = 16
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
        symmetric = ModelConfig.symmetric
        #
        import numpy as np
        from tensorflow import keras
        from tensorflow.keras import layers
        from qkeras import QConv2DBatchnorm, QActivation
        ### model start ###
        n_filter = 4
        conv_reg = keras.regularizers.l2(1e-3)
        #
        spec = keras.Input(shape=[3, 2, 1], dtype=np.float32)
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
        return [spec, x]

class SmallModelMicrofauneAI:
    def __init__(self):
        spec, pred = ModelConfig.customModel()
        #
        self.model = keras.Model(inputs=spec, outputs=pred)
        #
        self.model_quant = model_quantize(self.model, None, ModelConfig.bits, transfer_weights=True)
    #
    def modelOriginal(self):
        return self.model
    #
    def modelQuantized(self):
        return self.model_quant
