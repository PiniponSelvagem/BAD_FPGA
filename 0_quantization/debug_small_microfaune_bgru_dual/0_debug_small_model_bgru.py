import os
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from smallMicrofauneModel_bgru import SmallModelMicrofauneAI


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
model.weights

input_1
   >>> (None, 4, 2, 2)
   <<< (None, 4, 2, 2)
tf.math.reduce_max
   >>> (None, 4, 2, 2)
   <<< (None, 4, 2)
q_bidirectional
   >>> (None, 4, 2)
   <<< (None, 4, 4)


q_bidirectional/forward_qgru/qgru_cell_4/kernel:0
   --- (2, 6)
q_bidirectional/forward_qgru/qgru_cell_4/recurrent_kernel:0
   --- (2, 6)
q_bidirectional/forward_qgru/qgru_cell_4/bias:0
   --- (6,)
q_bidirectional/backward_qgru/qgru_cell_5/kernel:0
   --- (2, 6)
q_bidirectional/backward_qgru/qgru_cell_5/recurrent_kernel:0
   --- (2, 6)
q_bidirectional/backward_qgru/qgru_cell_5/bias:0
   --- (6,)
"""


#################################
# q_bidirectional__forward_qgru #
#################################
q_bidirectional__forward_qgru__kernel = np.array([[0.25, -0.25, 0.5, 0.5, 0.875, -0.875], [0.875, 0, 0, 0, 0, 0]])
q_bidirectional__forward_qgru__recurrent_kernel = np.array([[-0.375, 0.375, 0.625, -0.625, 0.5, -0.5], [0.875, 0, 0, 0, 0, 0]])
q_bidirectional__forward_qgru__bias = np.array([0.25, -0.25, 0.5, -0.5, 0.875, -0.875])
#
##################################
# q_bidirectional__backward_qgru #
##################################
q_bidirectional__backward_qgru__kernel = np.array([[0.375, -0.375, 0.625, 0.625, 0.875, -0.875], [0.875, 0, 0, 0, 0, 0]])
q_bidirectional__backward_qgru__recurrent_kernel = np.array([[0.875, 0, 0, 0, 0, 0], [0.875, 0, 0, 0, 0, 0]])
q_bidirectional__backward_qgru__bias = np.array([0, 0, 0, 0, 0, 0])


#####################################
# q_bidirectional_1__forward_qgru_1 #
#####################################
q_bidirectional_1__forward_qgru_1__kernel = np.array([[-0.25, 0.25, -0.5, 0.5, -0.875, 0.875], [0.875, 0.5, 0, 0, 0, 0], [-0.375, 0.375, -0.625, 0.625, -0.875, 0.875], [0.25, 0.125, 0, 0, 0, 0]])
q_bidirectional_1__forward_qgru_1__recurrent_kernel = np.array([[-0.125, 0.125, 0.625, -0.625, 0.25, -0.25], [0.875, -0.5, 0, 0, 0, 0]])
q_bidirectional_1__forward_qgru_1__bias = np.array([-0.25, 0.25, -0.5, 0.5, -0.875, 0.875])
#
######################################
# q_bidirectional_1__backward_qgru_1 #
######################################
q_bidirectional_1__backward_qgru_1__kernel = np.array([[0.25, 0.125, 0.375, -0.625, 0.25, -0.75], [0.125, -0.125, -0.625, -0.625, 0.25, 0.25], [0.5, -0.125, 0, 0, 0, 0], [0.25, 0, 0, 0, 0, 0]])
q_bidirectional_1__backward_qgru_1__recurrent_kernel = np.array([[0, 0.125, 0, 0, 0, 0], [0, 0, 0.5, 0, 0, 0]])
q_bidirectional_1__backward_qgru_1__bias = np.array([0, 0, 0.25, 0.5, 0, 0])




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
INPUT = 1
if INPUT == 0:
    # NOTE: THIS INPUT IS ONLY AFFECTED IN THIS SCRIPT
    #       TO CHANGE THE DUMP_IO INPUT, CHANGE ON THE OTHER SCRIPT "dump_io..."
    input = np.array([
        [
            [
                [ 1.0, 0 ],
                [ 0, 0 ]
            ],
            [
                [ 0,  0 ],
                [ 0,  0 ]
            ],
            [
                [ 0,  0 ],
                [ 0,  0 ]
            ],
            [
                [ 0,  0 ],
                [ 0,  0 ]
            ]
        ]
    ])
else:
    input = np.array([
        [
            [
                [ 1.0,  0.25 ],
                [ 0.25, 0.125]
            ],
            [
                [ 0.5,   0.75],
                [ 0.25,  0.875]
            ],
            [
                [ 0.375, 0.25 ],
                [ 0.625, 0.125]
            ],
            [
                [ 0.125,  0.5 ],
                [ 0.125,  0.75 ]
            ]
        ]
    ])
result = model.predict(input)
print(result)

