def dense():
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    #########################################################

    CUDA_VISIBLE_DEVICES=""     # force use CPU

    import numpy as np
    from tensorflow import keras
    from tensorflow.keras import layers

    spec = layers.Input(shape=[None, 4], dtype=np.float32)
    x = layers.Dense(2, activation="linear")(spec)
    model = keras.Model(inputs=spec, outputs=[x])

    kernel = [ [1, 2], [3, 4], [5, 6], [7, 8] ]
    bias = [ 1, 2 ]

    kernel = np.array(kernel)
    bias = np.array(bias)

    model.set_weights([kernel, bias])

    input = [ [ [ 6, 7, 8, 9 ], [ 1, 2, 3, 4 ] ] ]

    input = np.array(input)

    predict = model.predict(input)
    print(predict)


def td_dense():
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    #########################################################

    CUDA_VISIBLE_DEVICES=""     # force use CPU

    import numpy as np
    from tensorflow import keras
    from tensorflow.keras import layers

    spec = layers.Input(shape=[None, 4], dtype=np.float32)
    x = layers.TimeDistributed(layers.Dense(2, activation="linear"))(spec)
    model = keras.Model(inputs=spec, outputs=[x])

    kernel = [ [1, 2], [3, 4], [5, 6], [7, 8] ]
    bias = [ 1, 2 ]

    kernel = np.array(kernel)
    bias = np.array(bias)

    model.set_weights([kernel, bias])

    input = [ [ [ 6, 7, 8, 9 ], [ 1, 2, 3, 4 ] ] ]

    input = np.array(input)

    predict = model.predict(input)
    print(predict)



dense()
td_dense()