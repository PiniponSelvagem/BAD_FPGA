class ModelConfig:
    ##################################################
    #               !!! WARNING !!!                  #
    # MAKE SURE 'name' IS UNIQUE, OR DURING TRAINING #
    # IT WILL REPLACE AN OTHER MODEL WITH SAME NAME. #
    ##################################################
    name = "model_quant_411_noQuantState"    # model file name
    folder = "model_quantized"               # model saved location

    use_custom_model = False            # if False use microfaune original model, if True use customModel
    training_dataset_percentage = 1.0   # range [0.0, 1.0]

    bits = 4
    integer = 1
    symmetric = 1
    padding = "same"
    enable_bn_folding = False    # merge Conv with BNorm

    epochs = 40
    steps_per_epoch = 100

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
            #"state_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
            "kernel_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
            "recurrent_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
            "bias_quantizer": f"quantized_bits({bits},{integer},{symmetric})",
        },
    }