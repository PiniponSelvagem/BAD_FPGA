import numpy as np

#input
input = 1.1

#conv2d weights
kernel = 0.5
bias = 1

#batch_normalization weights
EPSILON = 0.001
gamma = 0.1
beta = 0.2
movingvariance = 0.3
movingmean = 0.4

#conv2d
output_conv2d = (input * kernel) + bias
print("original conv2d output: "+output_conv2d)
#batch_normalization
normalized = (output_conv2d - movingmean) / np.sqrt(movingvariance + EPSILON)
output = gamma * normalized + beta
print("original bnorm output: "+output)

# merged weights: conv2d <-> batch_normalization
W_merged = kernel / np.sqrt(movingvariance + EPSILON) * gamma
B_merged = (bias - movingmean) / np.sqrt(movingvariance) * gamma + beta

# using merged weights
inout = (input * W_merged) + B_merged
print("merged conv2d+bnorm output: "+inout)


"""
EPSILON = 0.001
merged_kernel = np.zeros_like(conv2d_kernel)
merged_bias = np.zeros_like(conv2d_bias)
for f in range(conv2d_kernel.shape[0]):
    for c in range(conv2d_kernel.shape[1]):
        for x in range(conv2d_kernel.shape[2]):
            for y in range(conv2d_kernel.shape[3]):
                merged_kernel[f, c, x, y] = conv2d_kernel[f, c, x, y] / np.sqrt(bn_moving_variance[f] + EPSILON) * bn_gamma[f]
    merged_bias[f] = (conv2d_bias[f] - bn_moving_mean[f]) / np.sqrt(bn_moving_variance[f] + EPSILON) * bn_gamma[f] + bn_beta[f]
"""