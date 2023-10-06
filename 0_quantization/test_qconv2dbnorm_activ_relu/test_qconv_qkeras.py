
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from tensorflow import keras
from qkeras import QActivation

from qkeras.utils import model_quantize

import matplotlib.pyplot as plt


bits = "4"
integer = "1"

### model start ###
spec = keras.Input(shape=[1], dtype=np.float32)
#
x = QActivation(f"quantized_relu({bits},{integer})")(spec)
### model end ###

_model = keras.Model(inputs=spec, outputs=x)
model = model_quantize(_model, None, bits, transfer_weights=True)



"""
OFFSET_QUANT = 0.0625
STEP_SIZE_QUANT = 0.125
MIN_QUANT = 0
MAX_QUANT = 1.875

def valueQuant(value):
    value = value + OFFSET_QUANT
    if value <= MIN_QUANT:
        return MIN_QUANT
    elif value >= MAX_QUANT:
        return MAX_QUANT
    step = int((value + 1) / STEP_SIZE_QUANT)
    return (step * STEP_SIZE_QUANT) - 1.0
"""



start_value = 0
end_value = 2
step = 0.00025

results = []

printProgress = False
total_iterations = int((end_value - start_value) / step) + 1
for i, value in enumerate(np.arange(start_value, end_value + step, step)):
    input_data = np.array([[value]])
    result = model.predict(input_data, verbose=0)
    resPred = result[0][0]
    results.append([value, resPred])
    #
    #resQuant = valueQuant(value)
    #if resPred != resQuant:
    #    print("input: "+str(value)+", output: "+str(resPred)+", customQuant:"+str(resQuant))
    #
    # Calculate the percentage of progress
    percentage = (i + 1) / total_iterations * 100
    #
    if printProgress:
        print(f"Progress: {percentage:.2f}%")


inputs = [result[0] for result in results]
outputs = [result[1] for result in results]

data_to_save = np.column_stack((inputs, outputs))
np.savetxt("results.csv", data_to_save, delimiter=',', header='Input,Output', comments='', fmt='%f')


#plt.clf()
plt.figure(figsize=(12, 8))
plt.plot(inputs, outputs)
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Input vs. Output")
plt.grid(True)

custom_xticks = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875]
plt.xticks(custom_xticks, custom_xticks)
custom_yticks = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875]
plt.yticks(custom_yticks, custom_yticks)

plt.savefig("output_plot.png")

