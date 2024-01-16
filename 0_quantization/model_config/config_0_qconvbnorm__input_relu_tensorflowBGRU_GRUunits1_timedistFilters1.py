class ModelConfig:
    ##################################################
    #               !!! WARNING !!!                  #
    # MAKE SURE 'name' IS UNIQUE, OR DURING TRAINING #
    # IT WILL REPLACE AN OTHER MODEL WITH SAME NAME. #
    ##################################################
    name = "model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1_timedistFilters1" # model file name
    folder = "model_quantized"          # model saved location

    use_custom_model = True             # if False use microfaune original model, if True use customModel
    training_dataset_percentage = 1.0   # range [0.0, 1.0]

    bits = 4
    integer = 1
    integer_relu = 0
    integer_gru_recact = 0
    symmetric = 1
    padding = "same"
    enable_bn_folding = True    # merge Conv with BNorm

    epochs = 100
    steps_per_epoch = 100

    
    config = { }

    ################
    # Custom Model #
    ################
    def customModel():
        bits = ModelConfig.bits
        integer = ModelConfig.integer
        integer_relu = ModelConfig.integer_relu
        integer_gru_recact = ModelConfig.integer_gru_recact
        symmetric = ModelConfig.symmetric
        gru_units = 1
        td_filter = gru_units
        #
        import numpy as np
        from tensorflow import keras
        from tensorflow.keras import layers
        from keras.layers import GRU
        from tensorflow import math
        from qkeras import QConv2DBatchnorm, QActivation, QBidirectional
        ### model start ###
        n_filter = 64
        conv_reg = keras.regularizers.l2(1e-3)
        #
        spec = keras.Input(shape=[431, 40, 1], dtype=np.float32)
        x = QActivation(f"quantized_relu({bits},{integer_relu})")(spec)
        #
        x = QConv2DBatchnorm(
            n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None,
            kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            momentum=0.95
        )(x)
        x = QActivation(f"quantized_relu({bits},{integer_relu})")(x)
        #
        x = QConv2DBatchnorm(
            n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None,
            kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            momentum=0.95
        )(x)
        x = QActivation(f"quantized_relu({bits},{integer_relu})")(x)
        #
        x = layers.MaxPool2D((1, 2))(x)
        #
        x = QConv2DBatchnorm(
            n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None,
            kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            momentum=0.95
        )(x)
        x = QActivation(f"quantized_relu({bits},{integer_relu})")(x)
        #
        x = QConv2DBatchnorm(
            n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None,
            kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            momentum=0.95
        )(x)
        x = QActivation(f"quantized_relu({bits},{integer_relu})")(x)
        #
        x = layers.MaxPool2D((1, 2))(x)
        #
        x = QConv2DBatchnorm(
            n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None,
            kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            momentum=0.95
        )(x)
        x = QActivation(f"quantized_relu({bits},{integer_relu})")(x)
        #
        x = QConv2DBatchnorm(
            n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None,
            kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
            momentum=0.95
        )(x)
        x = QActivation(f"quantized_relu({bits},{integer_relu})")(x)
        #
        if ModelConfig.padding == "same":
            x = layers.MaxPool2D((1, 2))(x)
        #
        x = math.reduce_max(x, axis=-2)
        #
        x = QBidirectional(GRU(gru_units, return_sequences=True))(x)
        x = QBidirectional(GRU(gru_units, return_sequences=True))(x)
        #
        x = layers.TimeDistributed(layers.Dense(td_filter, activation="sigmoid"))(x)
        local_pred = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
        pred = math.reduce_max(local_pred, axis=-2)
        #
        #
        return [spec, pred, local_pred]

