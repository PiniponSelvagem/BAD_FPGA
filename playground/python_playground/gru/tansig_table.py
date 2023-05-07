import os

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#########################################################

import numpy as np

end = 8
n = 201
step = (end - 0) / (n - 1)
columns_per_line = 5    # in the saved file
decimal_places = 16   # float->6, double->18

tansig_table = tf.linspace(0, end, n)
tansig_table = tf.tanh(tansig_table)
#np.savetxt('tansig_table.txt', tansig_table)

tansig_table = tansig_table.numpy().reshape(1,-1)

with open('tansig_table_python.txt', 'w') as f:
    for i, x in enumerate(tansig_table.flatten()):
        if i % columns_per_line == columns_per_line - 1 and i != n - 1:
            f.write('%.*f,\n' % (decimal_places, x))
        else:
            f.write('%.*f, ' % (decimal_places, x))

print(tansig_table)