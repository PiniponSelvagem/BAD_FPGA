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
   >>> (None, 3, 1)
   <<< (None, 3, 1)
q_activation
   >>> (None, 3, 1)
   <<< (None, 3, 1)
q_bidirectional
   >>> (None, 3, 1)
   <<< (None, 3, 8)
q_bidirectional_1
   >>> (None, 3, 8)
   <<< (None, 3, 8)
"""
"""
Model layers WEIGHTS (shape)
q_bidirectional/forward_qgru/qgru_cell_7/kernel:0
   --- (1, 12)
q_bidirectional/forward_qgru/qgru_cell_7/recurrent_kernel:0
   --- (4, 12)
q_bidirectional/forward_qgru/qgru_cell_7/bias:0
   --- (12,)
q_bidirectional/backward_qgru/qgru_cell_8/kernel:0
   --- (1, 12)
q_bidirectional/backward_qgru/qgru_cell_8/recurrent_kernel:0
   --- (4, 12)
q_bidirectional/backward_qgru/qgru_cell_8/bias:0
   --- (12,)
q_bidirectional_1/forward_qgru_1/qgru_cell_10/kernel:0
   --- (8, 12)
q_bidirectional_1/forward_qgru_1/qgru_cell_10/recurrent_kernel:0
   --- (4, 12)
q_bidirectional_1/forward_qgru_1/qgru_cell_10/bias:0
   --- (12,)
q_bidirectional_1/backward_qgru_1/qgru_cell_11/kernel:0
   --- (8, 12)
q_bidirectional_1/backward_qgru_1/qgru_cell_11/recurrent_kernel:0
   --- (4, 12)
q_bidirectional_1/backward_qgru_1/qgru_cell_11/bias:0
   --- (12,)
"""


############################################################################
# Explanation of GRU weights order in TensorFlow and QKeras for this model #
############################################################################
"""
Example for a model with 2 filters:
- Legend:
--- 0 -> 1st filter, 1 -> 2nd filter, etc...)

GRU_0:
- Forward and Backward GRU:
kernel = [                  rkernel = [
    z_0, z_1,                   state_0 -> [ rz_0, rz_1, rr_0, rr_1, rh_0, rh_1 ],
    r_0, r_1,                   state_1 -> [ rz_0, rz_1, rr_0, rr_1, rh_0, rh_1 ]
    h_0, h_1                ]
]

GRU_1:
- Forward and Barckward:
kernel = [                  rkernel = [
    z_0, z_1, z_2, z_3,         state_0 -> [ rz_0, rz_1, rr_0, rr_1, rh_0, rh_1 ],
    r_0, r_1, r_2, r_3,         state_1 -> [ rz_0, rz_1, rr_0, rr_1, rh_0, rh_1 ]
    h_0, h_1, h_2, h_3      ]
]

"""


#################################
# q_bidirectional__forward_qgru #
#################################
q_bidirectional__forward_qgru__kernel = np.array([[0.125,-0.125, 0.25,-0.25, 0.5,-0.5]])
q_bidirectional__forward_qgru__recurrent_kernel = np.array(
    [
        [0.4375,-0.4375, 0.5625,-0.5625, 0.3125,-0.3125],
        [0.5,-0.5, 0.0625,-0.625, 1,-1]
    ]
)
q_bidirectional__forward_qgru__bias = np.array([0,0, 0,0, 0,0])
#
##################################
# q_bidirectional__backward_qgru #
##################################
q_bidirectional__backward_qgru__kernel = np.array([[0.0625,-0.0625, 0.75,-0.75, 0.875,-0.875]])
q_bidirectional__backward_qgru__recurrent_kernel = np.array(
    [
        [0.5,-0.5, 0.5625,-0.5625, 0.3125,-0.3125],
        [0.4375,-0.4375, 1,-1, 0.0625,-0.625]
    ]
)
q_bidirectional__backward_qgru__bias = np.array([0,0, 0,0, 0,0])


#####################################
# q_bidirectional_1__forward_qgru_1 #
#####################################
q_bidirectional_1__forward_qgru_1__kernel = np.array(
    [
        [1,0, 1,0, 1,0],
        [1,0, 1,0, 1,0],
        [1,0, 1,0, 1,0],
        [1,0, 1,0, 1,0]
    ]
)
q_bidirectional_1__forward_qgru_1__recurrent_kernel = np.array(
    [
        [1,0, 1,0, 1,0],
        [1,0, 1,0, 1,0]
    ]
)
q_bidirectional_1__forward_qgru_1__bias = np.array([0,0, 0,0, 0,0])
#
######################################
# q_bidirectional_1__backward_qgru_1 #
######################################
q_bidirectional_1__backward_qgru_1__kernel = np.array(
    [
        [1,0, 1,0, 1,0],
        [1,0, 1,0, 1,0],
        [1,0, 1,0, 1,0],
        [1,0, 1,0, 1,0]
    ]
)
q_bidirectional_1__backward_qgru_1__recurrent_kernel = np.array(
    [
        [1,0, 1,0, 1,0],
        [1,0, 1,0, 1,0]
    ]
)
q_bidirectional_1__backward_qgru_1__bias = np.array([0,0, 0,0, 0,0])


"""
#################################
# q_bidirectional__forward_qgru #
#################################
q_bidirectional__forward_qgru__kernel = np.array([[1,0,0,0, 1,0,0,0, 1,0,0,0]])
q_bidirectional__forward_qgru__recurrent_kernel = np.array(
    [
        [-1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0]
    ]
)
q_bidirectional__forward_qgru__bias = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1])
#
##################################
# q_bidirectional__backward_qgru #
##################################
q_bidirectional__backward_qgru__kernel = np.array([[0.23,0,0,0, 0.24,0,0,0, 0.25,0,0,0]])
q_bidirectional__backward_qgru__recurrent_kernel = np.array(
    [
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [-1,0,0,0, 1,0,0,0, 1,0,0,0]
    ]
)
q_bidirectional__backward_qgru__bias = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1])


#####################################
# q_bidirectional_1__forward_qgru_1 #
#####################################
q_bidirectional_1__forward_qgru_1__kernel = np.array(
    [
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0]
    ]
)
q_bidirectional_1__forward_qgru_1__recurrent_kernel = np.array(
    [
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0]
    ]
)
q_bidirectional_1__forward_qgru_1__bias = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1])
#
######################################
# q_bidirectional_1__backward_qgru_1 #
######################################
q_bidirectional_1__backward_qgru_1__kernel = np.array(
    [
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0]
    ]
)
q_bidirectional_1__backward_qgru_1__recurrent_kernel = np.array(
    [
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0],
        [1,0,0,0, 1,0,0,0, 1,0,0,0]
    ]
)
q_bidirectional_1__backward_qgru_1__bias = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1])
"""



model.set_weights([
    q_bidirectional__forward_qgru__kernel, q_bidirectional__forward_qgru__recurrent_kernel, q_bidirectional__forward_qgru__bias,
    q_bidirectional__backward_qgru__kernel, q_bidirectional__backward_qgru__recurrent_kernel, q_bidirectional__backward_qgru__bias,
    q_bidirectional_1__forward_qgru_1__kernel, q_bidirectional_1__forward_qgru_1__recurrent_kernel, q_bidirectional_1__forward_qgru_1__bias,
    q_bidirectional_1__backward_qgru_1__kernel, q_bidirectional_1__backward_qgru_1__recurrent_kernel, q_bidirectional_1__backward_qgru_1__bias,
])

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.save_weights(f"{save_dir}/{model_name}.h5")

#####################################################################################################################


#######################
# INPUT and inference #
#######################
input = np.array(
[   # NOTE: THIS INPUT IS ONLY AFFECTED IN THIS SCRIPT
    #       TO CHANGE THE DUMP_IO INPUT, CHANGE ON THE OTHER SCRIPT "dump_io..."
    [
        [[0.625]],
        [[0.25]],
        [[0.3125]],
    ]
])

result = model.predict(input)
print(result)

