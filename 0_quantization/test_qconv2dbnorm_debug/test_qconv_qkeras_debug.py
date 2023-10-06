
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from qkeras.utils import model_quantize
### model start ###
bits = 16
integer = 4
symmetric = 1
#
import numpy as np
from tensorflow import keras
from qkeras import QConv2DBatchnorm
### model start ###
n_filter = 2
conv_reg = keras.regularizers.l2(1e-3)
#
spec = keras.Input(shape=[4, 4, 2], dtype=np.float32)
x = QConv2DBatchnorm(
    n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None,
    kernel_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
    bias_quantizer = f"quantized_bits({bits},{integer},{symmetric})",
    momentum=0.95
)(spec)
### model end ###
_qmodel = keras.Model(inputs=spec, outputs=x)
qmodel = model_quantize(_qmodel, None, bits, transfer_weights=True)



from tensorflow.keras import layers
### model start ###
x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(spec)
x = layers.BatchNormalization(momentum=0.95)(x)
### model end ###
model = keras.Model(inputs=spec, outputs=x)


gamma = np.array([1., 1.])
beta  = np.array([0., 0.])
moving_mean = np.array([0., 0.])
moving_variance = np.array([0., 0.])


kernel = np.array(
    [
        [
            [
                [0.125,  -0.125],
                [0.25,   -0.25 ]
            ],
            [
                [0.375,  -0.375],
                [0.5,    -0.5  ]
            ],
            [
                [0.625,  -0.625],
                [0.75,   -0.75 ]
            ]
        ],
        [
            [
                [-0.875,  0.875],
                [0.0,     0.0  ]
            ],
            [
                [0.25,   -0.25 ],
                [0.75,   -0.75 ]
            ],
            [
                [0.25,   -0.25 ],
                [0.5,    -0.5  ]
            ]
        ],
        [
            [
                [0.0,     0.0  ],
                [0.0,     0.0  ]
            ],
            [
                [0.0,     0.0  ],
                [0.0,     0.0  ]
            ],
            [
                [0.25,   -0.25 ],
                [0.5,    -0.5  ]
            ]
        ]
    ]
)
bias = np.array([-0.5, 0.5])

qmodel.set_weights([kernel, bias, gamma, beta, 1, moving_mean, moving_variance])
model.set_weights([kernel, bias, gamma, beta, moving_mean, moving_variance])

input = np.array(
    [
        [
            [
                [0.1, -0.2], [-0.3, 0.4], [-0.1, 0.2], [0.3, -0.4]
            ],
            [
                [-0.2, 0.1], [0.4, -0.3], [0.2, -0.1], [-0.4, 0.3]
            ],
            [
                [0.1, -0.2], [-0.3, 0.4], [-0.1, 0.2], [0.3, -0.4]
            ],
            [
                [-0.2, 0.1], [0.4, -0.3], [0.2, -0.1], [-0.4, 0.3]
            ]
        ]
    ]
)


outQModel = qmodel.predict(input)
outModel = model.predict(input)

outQModel
outModel

