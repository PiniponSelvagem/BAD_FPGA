import os
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from smallMicrofauneModel import SmallModelMicrofauneAI


save_dir = "model_debug"
model_name = "smallModel"

model = SmallModelMicrofauneAI().modelQuantized()

print("\nModel layers INPUT / OUTPUT (shape)")
for l in model.layers:
    print(str(l.name) +"\n   >>> "+ str(l.input.shape) +"\n   <<< "+ str(l.output.shape))

print("\nModel layers WEIGHTS (shape)")
for l in model.weights:
    print(str(l.name) +"\n   --- "+ str(l.shape))


#####################################################################################################################

"""
Model layers INPUT / OUTPUT (shape)
input_1
   >>> (None, 3, 2, 1)
   <<< (None, 3, 2, 1)
q_conv2d_batchnorm
   >>> (None, 3, 2, 1)
   <<< (None, 3, 2, 16)
q_activation
   >>> (None, 3, 2, 16)
   <<< (None, 3, 2, 16)
q_conv2d_batchnorm_1
   >>> (None, 3, 2, 16)
   <<< (None, 3, 2, 16)
q_activation_1
   >>> (None, 3, 2, 16)
   <<< (None, 3, 2, 16)
max_pooling2d
   >>> (None, 3, 2, 16)
   <<< (None, 3, 1, 16)
"""
"""
Model layers WEIGHTS (shape)
q_conv2d_batchnorm/kernel:0
   --- (3, 3, 1, 16)
q_conv2d_batchnorm/bias:0
   --- (16,)
q_conv2d_batchnorm/batch_normalization_2/gamma:0
   --- (16,)
q_conv2d_batchnorm/batch_normalization_2/beta:0
   --- (16,)
q_conv2d_batchnorm/iteration:0
   --- ()
q_conv2d_batchnorm/batch_normalization_2/moving_mean:0
   --- (16,)
q_conv2d_batchnorm/batch_normalization_2/moving_variance:0
   --- (16,)
q_conv2d_batchnorm_1/kernel:0
   --- (3, 3, 16, 16)
q_conv2d_batchnorm_1/bias:0
   --- (16,)
q_conv2d_batchnorm_1/batch_normalization_3/gamma:0
   --- (16,)
q_conv2d_batchnorm_1/batch_normalization_3/beta:0
   --- (16,)
q_conv2d_batchnorm_1/iteration:0
   --- ()
q_conv2d_batchnorm_1/batch_normalization_3/moving_mean:0
   --- (16,)
q_conv2d_batchnorm_1/batch_normalization_3/moving_variance:0
   --- (16,)
"""

######################
# q_conv2d_batchnorm #
######################
q_conv2d_batchnorm__kernel = np.array(
    [
        [   [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]], [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]], [[-1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]]   ],
        [   [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]], [[-8,-7,-6,-5, -4,-3,-2,-1, 0,1,2,3, 4,5,6,7]], [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]]   ],
        [   [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]], [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]], [[-3.5,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]]   ]
    ]
)
q_conv2d_batchnorm__bias = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1]) #1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4
#
q_conv2d_batchnorm__gamma = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])
q_conv2d_batchnorm__beta = np.array([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0])
q_conv2d_batchnorm__iteration = np.array(-1)
q_conv2d_batchnorm__moving_mean = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])
q_conv2d_batchnorm__moving_variance = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])


########################
# q_conv2d_batchnorm_1 #
########################
q_conv2d_batchnorm_1__kernel = np.array(
    [
        [
            [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]],
            [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]],
            [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]]
        ],
        [
            [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]],
            [[1,0,0,2, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0.75,0,0,2, 0,0,0,0, 0,0,0,0, 0,0,0,0], [1,0,0,2, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]],
            [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]]
        ],
        [
            [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]],
            [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]],
            [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]]
        ]
    ]
)
q_conv2d_batchnorm_1__bias = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1]) #1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4
#
q_conv2d_batchnorm_1__gamma = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])
q_conv2d_batchnorm_1__beta = np.array([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0])
q_conv2d_batchnorm_1__iteration = np.array(-1)
q_conv2d_batchnorm_1__moving_mean = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])
q_conv2d_batchnorm_1__moving_variance = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])




model.set_weights([
    q_conv2d_batchnorm__kernel, q_conv2d_batchnorm__bias, q_conv2d_batchnorm__gamma, q_conv2d_batchnorm__beta, q_conv2d_batchnorm__iteration, q_conv2d_batchnorm__moving_mean, q_conv2d_batchnorm__moving_variance,
    q_conv2d_batchnorm_1__kernel, q_conv2d_batchnorm_1__bias, q_conv2d_batchnorm_1__gamma, q_conv2d_batchnorm_1__beta, q_conv2d_batchnorm_1__iteration, q_conv2d_batchnorm_1__moving_mean, q_conv2d_batchnorm_1__moving_variance,
])

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.save_weights(f"{save_dir}/{model_name}.h5")

#####################################################################################################################


#######################
# INPUT and inference #
#######################
INPUT = 0
if INPUT == 0:
    input = np.array(
    [   # NOTE: THIS INPUT IS ONLY AFFECTED IN THIS SCRIPT
        #       TO CHANGE THE DUMP_IO INPUT, CHANGE ON THE OTHER SCRIPT "dump_io..."
        [
            [[0.125],[0.25]],
            [[0.375],[0.5 ]],
            [[0.625],[0.75]],
        ]
    ])
else:
    input = np.array(
    [
        [
            [   [1],    [0.5],  [1]     ],
            [   [0.25], [0.25], [0.5]   ],
            [   [0.75], [1],    [-0.25] ],
            [   [-1],   [0],    [0.5]   ]
        ]
    ])

result = model.predict(input)
print(result)

