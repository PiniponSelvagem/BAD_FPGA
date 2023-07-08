input  = [[1, 2], [3, 4], [8, 9], [10, 11]]

kernel = [[5, 6], [7, 8]]
bias   = [1, 2]


output = [[0, 0], [0, 0], [0, 0], [0, 0]]

output[0][0] = (input[0][0] * kernel[0][0] + input[0][1] * kernel[1][0]) + bias[0]
output[0][1] = (input[0][0] * kernel[0][1] + input[0][1] * kernel[1][1]) + bias[1]
output[1][0] = (input[1][0] * kernel[0][0] + input[1][1] * kernel[1][0]) + bias[0]
output[1][1] = (input[1][0] * kernel[0][1] + input[1][1] * kernel[1][1]) + bias[1]

output[2][0] = (input[2][0] * kernel[0][0] + input[2][1] * kernel[1][0]) + bias[0]
output[2][1] = (input[2][0] * kernel[0][1] + input[2][1] * kernel[1][1]) + bias[1]
output[3][0] = (input[3][0] * kernel[0][0] + input[3][1] * kernel[1][0]) + bias[0]
output[3][1] = (input[3][0] * kernel[0][1] + input[3][1] * kernel[1][1]) + bias[1]

for row in output:
    print(row)
print()


###############################

import numpy as np

print("numpy.dot (does not have bias)")
outnp = np.dot(input, kernel)
print(outnp)
