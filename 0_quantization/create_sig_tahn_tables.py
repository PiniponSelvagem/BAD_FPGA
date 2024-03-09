import os
import numpy as np
import cutils


folder = "lookup_tables"


    # SIGMOID #
W_SIG = 8
I_SIG = 0
SIG_TABLE_SIZE = 256

SIG_startValue_F = -6.3
SIG_endValue_F   =  5.2
    ###########

    # TAHN #
W_TANH = 8
I_TANH = 1
TANH_TABLE_SIZE = 256

TANH_startValue_F = -2.6
TANH_endValue_F   =  2.6
    ########

###################################################

sigmoidTable = np.array([])
tanhTable = np.array([])

if not os.path.isdir(folder):
    os.makedirs(folder)




def zsigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def loadSigmoidTable():
    global sigmoidTable
    step = (SIG_endValue_F - SIG_startValue_F) / (SIG_TABLE_SIZE - 1)
    for i in range(0, SIG_TABLE_SIZE):
        x = SIG_startValue_F + i * step
        sigmoidTable = np.append(sigmoidTable, zsigmoid(x))

def loadTanhTable():
    global tanhTable
    step = (TANH_endValue_F - TANH_startValue_F) / (TANH_TABLE_SIZE - 1)
    for i in range(0, TANH_TABLE_SIZE):
        x = TANH_startValue_F + i * step
        tanhTable = np.append(tanhTable, np.tanh(x))

loadSigmoidTable()
loadTanhTable()

cutils.saveArray_dim1(folder, "sigmoid", sigmoidTable, "sigmoidTable", "float")
cutils.saveArray_dim1(folder, "tanh", tanhTable, "tanhTable", "float")
