class ModelConfig:
    ##################################################
    #               !!! WARNING !!!                  #
    # MAKE SURE 'name' IS UNIQUE, OR DURING TRAINING #
    # IT WILL REPLACE AN OTHER MODEL WITH SAME NAME. #
    ##################################################
    name = "model_quant_411_qconvbnorm" # model file name
    folder = "model_quantized"          # model saved location

    use_custom_model = True             # if False use microfaune original model, if True use customModel
    training_dataset_percentage = 1.0   # range [0.0, 1.0]

    bits = 4
    integer = 1
    integer_gru_recact = 0
    symmetric = 1
    padding = "same"
    enable_bn_folding = True    # merge Conv with BNorm

    epochs = 40
    steps_per_epoch = 100

    
    config = { }

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
        n_filter = 64
        conv_reg = keras.regularizers.l2(1e-3)
        #
        spec = keras.Input(shape=[431, 40, 1], dtype=np.float32)
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
        x = QConv2DBatchnorm(
            n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None,
            kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            momentum=0.95
        )(x)
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
        x = QConv2DBatchnorm(
            n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None,
            kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            momentum=0.95
        )(x)
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
        if ModelConfig.padding == "same":
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






"""

layer = dual_model.layers[1]

weights = []
qs = layer.get_quantizers()
ws = layer.get_folded_weights()

for quantizer, weight in zip(qs, ws):
    if quantizer:
        weight = tf.constant(weight)
        weight = tf.keras.backend.eval(quantizer(weight))
        weights.append(weight)


"""