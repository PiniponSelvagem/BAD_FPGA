class ModelConfig:
    ##################################################
    #               !!! WARNING !!!                  #
    # MAKE SURE 'name' IS UNIQUE, OR DURING TRAINING #
    # IT WILL REPLACE AN OTHER MODEL WITH SAME NAME. #
    ##################################################
    name = "model_sample"       # model file name
    folder = "model_quantized"  # model saved location

    use_custom_model = True             # if False use microfaune original model, if True use customModel
    training_dataset_percentage = 1.0   # range [0.0, 1.0]

    bits = 4
    integer = 1
    symmetric = 1
    padding = "same"
    enable_bn_folding = False    # merge Conv with BNorm

    epochs = 1
    steps_per_epoch = 1

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
            #"recurrent_activation": f"quantized_sigmoid({bits})",      # no effect????
            "state_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
            "kernel_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
            "recurrent_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
            "bias_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
        },
    }

    ################
    # Custom Model #
    ################
    def customModel():
        import numpy as np
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow import math
        ### model start ###
        n_filter = 64
        conv_reg = keras.regularizers.l2(1e-3)
        #
        spec = keras.Input(shape=[431, 40, 1], dtype=np.float32)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None)(spec)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.MaxPool2D((1, 2))(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.MaxPool2D((1, 2))(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding=ModelConfig.padding, kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        if ModelConfig.padding == "same":
            x = layers.MaxPool2D((1, 2))(x)
        #
        x = math.reduce_max(x, axis=-2)
        #
        x = layers.Bidirectional(layers.GRU(n_filter, return_sequences=True))(x)
        x = layers.Bidirectional(layers.GRU(n_filter, return_sequences=True))(x)
        #
        x = layers.TimeDistributed(layers.Dense(n_filter, activation="sigmoid"))(x)
        local_pred = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
        pred = math.reduce_max(local_pred, axis=-2)
        #
        #
        return [spec, pred, local_pred]