from tensorflow import keras
from qkeras.utils import model_quantize
from tensorflow import math


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
        from qkeras import QBidirectional, QGRU
        ### model start ###
        n_filter = 2
        #
        spec = keras.Input(shape=[4, 2, n_filter], dtype=np.float32)      # [431, 5, 64]
        #
        x = math.reduce_max(spec, axis=-2)
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
        return [spec, x]

class SmallModelMicrofauneAI:
    def __init__(self):
        spec, x = ModelConfig.customModel()
        #
        self.model = keras.Model(inputs=spec, outputs=x)
        #
        self.model_quant = model_quantize(self.model, None, ModelConfig.bits, transfer_weights=True)
    #    
    def modelOriginal(self):
        return self.model
    #
    def modelQuantized(self):
        return self.model_quant
