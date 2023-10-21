import os
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from smallMicrofauneModel import SmallModelMicrofauneAI


save_dir = "model_debug"
model_name = "smallModel"

_, model = SmallModelMicrofauneAI().modelQuantized()

print("\nModel layers INPUT / OUTPUT (shape)")
for l in model.layers:
    print(str(l.name) +"\n   >>> "+ str(l.input.shape) +"\n   <<< "+ str(l.output.shape))

print("\nModel layers WEIGHTS (shape)")
for l in model.weights:
    print(str(l.name) +"\n   --- "+ str(l.shape))


#####################################################################################################################

"""
model.weights

q_conv2d_batchnorm
    kernel -----------> shape = (3, 3, 1, 2)
    bias -------------> shape = (2)
    ------------------------------------------------------------------------------
    gamma ------------> shape = (2)         default = [1, 1]
    beta -------------> shape = (2)         default = [0, 0]
    iteration --------> shape = ()          ??? np.array() ???
    moving_mean ------> shape = (2)         default = [0, 0]
    moving_variance --> shape = (2)         default = [1, 1]

q_conv2d_batchnorm_1
    kernel -----------> shape = (3, 3, 2, 2)
    bias -------------> shape = (2)
    ------------------------------------------------------------------------------
    gamma ------------> shape = (2)         default = [1, 1]
    beta -------------> shape = (2)         default = [0, 0]
    iteration --------> shape = ()          ??? np.array() ???
    moving_mean ------> shape = (2)         default = [0, 0]
    moving_variance --> shape = (2)         default = [1, 1]

q_bidirectional/forward_qgru
    kernel -----------> shape = (2, 6)
    recurrent_kernel -> shape = (2, 6)
    bias -------------> shape = (6)
q_bidirectional/backward_qgru
    kernel -----------> shape = (2, 6)
    recurrent_kernel -> shape = (2, 6)
    bias -------------> shape = (6)

q_bidirectional_1/forward_qgru_1
    kernel -----------> shape = (4, 6)
    recurrent_kernel -> shape = (2, 6)
    bias -------------> shape = (6)
q_bidirectional_1/backward_qgru_1
    kernel -----------> shape = (4, 6)
    recurrent_kernel -> shape = (2, 6)
    bias -------------> shape = (6)

time_distributed
    kernel -----------> shape = (4, 2)
    bias -------------> shape = (2)

time_distributed_1
    kernel -----------> shape = (2, 1)
    bias -------------> shape = (1)
"""

######################
# q_conv2d_batchnorm #
######################
q_conv2d_batchnorm__kernel = np.array(
    [
        [   [[0, 0]], [[0, 0]], [[0, 0]]   ],
        [   [[0, 0]], [[1, 1]], [[0, 0]]   ],
        [   [[0, 0]], [[0, 0]], [[0, 0]]   ]
    ]
)
q_conv2d_batchnorm__bias = np.array([0, 0])
#
q_conv2d_batchnorm__gamma = np.array([1, 1])            # default (1,1)
q_conv2d_batchnorm__beta = np.array([0, 0])             # default (0,0)
q_conv2d_batchnorm__iteration = np.array(-1)            # default -1
q_conv2d_batchnorm__moving_mean = np.array([1, 1])      # default (0,0)
q_conv2d_batchnorm__moving_variance = np.array([1, 1])  # default (1,1)


########################
# q_conv2d_batchnorm_1 #
########################
q_conv2d_batchnorm_1__kernel = np.array(
    [
        [   [[0, 0], [0, 0]],   [[0, 0], [0, 0]],   [[0, 0], [0, 0]]   ],
        [   [[0, 0], [0, 0]],   [[1, 1], [1, 1]],   [[0, 0], [0, 0]]   ],
        [   [[0, 0], [0, 0]],   [[0, 0], [0, 0]],   [[0, 0], [0, 0]]   ]
    ]
)
q_conv2d_batchnorm_1__bias = np.array([0, 0])
#
q_conv2d_batchnorm_1__gamma = np.array([1, 1])            # default (1,1)
q_conv2d_batchnorm_1__beta = np.array([0, 0])             # default (0,0)
q_conv2d_batchnorm_1__iteration = np.array(-1)            # default -1
q_conv2d_batchnorm_1__moving_mean = np.array([1, 1])      # default (0,0)
q_conv2d_batchnorm_1__moving_variance = np.array([1, 1])  # default (1,1)


#################################
# q_bidirectional__forward_qgru #
#################################
q_bidirectional__forward_qgru__kernel = np.array([[0.25, -0.25, 0.5, 0.5, 1, -1], [1, 0, 0, 0, 0, 0]])
q_bidirectional__forward_qgru__recurrent_kernel = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
q_bidirectional__forward_qgru__bias = np.array([0.25, -0.25, 0.5, -0.5, 1, -1])
#
##################################
# q_bidirectional__backward_qgru #
##################################
q_bidirectional__backward_qgru__kernel = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
q_bidirectional__backward_qgru__recurrent_kernel = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
q_bidirectional__backward_qgru__bias = np.array([0, 0, 0, 0, 0, 0])


#####################################
# q_bidirectional_1__forward_qgru_1 #
#####################################
q_bidirectional_1__forward_qgru_1__kernel = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
q_bidirectional_1__forward_qgru_1__recurrent_kernel = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
q_bidirectional_1__forward_qgru_1__bias = np.array([0, 0, 0, 0, 0, 0])
#
######################################
# q_bidirectional_1__backward_qgru_1 #
######################################
q_bidirectional_1__backward_qgru_1__kernel = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
q_bidirectional_1__backward_qgru_1__recurrent_kernel = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
q_bidirectional_1__backward_qgru_1__bias = np.array([0, 0, 0, 0, 0, 0])


####################
# time_distributed #
####################
time_distributed__kernel = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
time_distributed__bias = np.array([0, 0])


######################
# time_distributed_1 #
######################
time_distributed_1__kernel = np.array([[1], [1]])
time_distributed_1__bias = np.array([0])


model.set_weights([
    q_conv2d_batchnorm__kernel, q_conv2d_batchnorm__bias, q_conv2d_batchnorm__gamma, q_conv2d_batchnorm__beta, q_conv2d_batchnorm__iteration, q_conv2d_batchnorm__moving_mean, q_conv2d_batchnorm__moving_variance,
    q_conv2d_batchnorm_1__kernel, q_conv2d_batchnorm_1__bias, q_conv2d_batchnorm_1__gamma, q_conv2d_batchnorm_1__beta, q_conv2d_batchnorm_1__iteration, q_conv2d_batchnorm_1__moving_mean, q_conv2d_batchnorm_1__moving_variance,
    q_bidirectional__forward_qgru__kernel, q_bidirectional__forward_qgru__recurrent_kernel, q_bidirectional__forward_qgru__bias,
    q_bidirectional__backward_qgru__kernel, q_bidirectional__backward_qgru__recurrent_kernel, q_bidirectional__backward_qgru__bias,
    q_bidirectional_1__forward_qgru_1__kernel, q_bidirectional_1__forward_qgru_1__recurrent_kernel, q_bidirectional_1__forward_qgru_1__bias,
    q_bidirectional_1__backward_qgru_1__kernel, q_bidirectional_1__backward_qgru_1__recurrent_kernel, q_bidirectional_1__backward_qgru_1__bias,
    time_distributed__kernel, time_distributed__bias,
    time_distributed_1__kernel, time_distributed_1__bias
])

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.save_weights(f"{save_dir}/{model_name}.h5")

#####################################################################################################################


#######################
# INPUT and inference #
#######################
INPUT = 1
if INPUT == 0:
    input = np.array(
    [   # NOTE: THIS INPUT IS ONLY AFFECTED IN THIS SCRIPT
        #       TO CHANGE THE DUMP_IO INPUT, CHANGE ON THE OTHER SCRIPT "dump_io..."
        [
            [[1],[0],[0]],
            [[0],[0],[0]],
            [[0],[0],[0]],
            [[0],[0],[0]]
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

