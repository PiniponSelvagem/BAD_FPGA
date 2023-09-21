class ModelConfig:
    ##################################################
    #               !!! WARNING !!!                  #
    # MAKE SURE 'name' IS UNIQUE, OR DURING TRAINING #
    # IT WILL REPLACE AN OTHER MODEL WITH SAME NAME. #
    ##################################################
    name = "model_test"         # model file name
    folder = "model_quantized"  # model saved location

    bits = 4
    integer = 1
    symetric = 1
    padding = "same"

    epochs = 1
    steps_per_epoch = 1

    config = {
        "QConv2D": {
            "kernel_quantizer": f"quantized_bits({bits},{integer},{symetric})",
            "bias_quantizer": f"quantized_bits({bits},{integer},{symetric})",
        },
        "QBatchNormalization": {
            "mean_quantizer": f"quantized_bits({bits},{integer},{symetric})",
            "gamma_quantizer": f"quantized_bits({bits},{integer},{symetric})",
            "variance_quantizer": f"quantized_bits({bits},{integer},{symetric})",
            "beta_quantizer": f"quantized_bits({bits},{integer},{symetric})",
            "inverse_quantizer": f"quantized_bits({bits},{integer},{symetric})",
        },
        "QBidirectional": {
            "activation": f"quantized_tanh({bits})",
            #"recurrent_activation": f"quantized_sigmoid({bits})",      # no effect????
            "state_quantizer": f"quantized_bits({bits},{integer},{symetric})",
            "kernel_quantizer": f"quantized_bits({bits},{integer},{symetric})",
            "recurrent_quantizer": f"quantized_bits({bits},{integer},{symetric})",
            "bias_quantizer": f"quantized_bits({bits},{integer},{symetric})",
        },
    }