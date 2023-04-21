import numpy as np

def gru_old(input, f_kernel, f_r_kernel, f_bias, b_kernel, b_r_kernel, b_bias):
    timesteps, input_dim = len(input), len(input[0])
    units = len(f_kernel[0])
    
    # Initialize hidden state
    h_t = [[0.0] * units]
    
    for t in range(timesteps):
        x_t = input[t]
        z_t = [[0.0] * units]
        r_t = [[0.0] * units]
        h_tilde_t = [[0.0] * units]
        for j in range(units):
            z_t[j] = f_bias[j]
            for i in range(input_dim):
                print("z_t ", z_t[j], "\nx_t ", x_t[i], "\nf_kernel ", f_kernel[i][j])
                z_t[j] += x_t[i] * f_kernel[i][j]
                z_t[j] += h_t[i] * f_r_kernel[i][j]
            z_t[j] = 1 / (1 + np.exp(-z_t[j]))
            #
            r_t[j] = f_bias[j]
            for i in range(input_dim):
                r_t[j] += x_t[i] * f_kernel[i][j]
                r_t[j] += h_t[i] * f_r_kernel[i][j]
            r_t[j] = 1 / (1 + np.exp(-r_t[j]))
            #
            h_tilde_t[j] = f_bias[j]
            for i in range(input_dim):
                h_tilde_t[j] += x_t[i] * f_kernel[i][j]
                h_tilde_t[j] += (r_t[j] * h_t[i]) * f_r_kernel[i][j]
            h_tilde_t[j] = np.tanh(h_tilde_t[j])
        #
        for j in range(units):
            h_t[j] = (1 - z_t[j]) * h_t[j] + z_t[j] * h_tilde_t[j]
        #
        x_t = input[-t-1]
        z_t = [[0.0] * units]
        r_t = [[0.0] * units]
        h_tilde_t = [[0.0] * units]
        for j in range(units):
            z_t[j] = b_bias[j]
            for i in range(input_dim):
                z_t[j] += x_t[i] * b_kernel[i][j]
                z_t[j] += h_t[i] * b_r_kernel[i][j]
            z_t[j] = 1 / (1 + np.exp(-z_t[j]))
            #
            r_t[j] = b_bias[j]
            for i in range(input_dim):
                r_t[j] += x_t[i] * b_kernel[i][j]
                r_t[j] += h_t[i] * b_r_kernel[i][j]
            r_t[j] = 1 / (1 + np.exp(-r_t[j]))
            #
            h_tilde_t[j] = b_bias[j]
            for i in range(input_dim):
                h_tilde_t[j] += x_t[i] * b_kernel[i][j]
                h_tilde_t[j] += (r_t[j] * h_t[i]) * b_r_kernel[i][j]
            h_tilde_t[j] = np.tanh(h_tilde_t[j])
        #
        for j in range(units):
            h_t[j] = (1 - z_t[j]) * h_t[j] + z_t[j] * h_tilde_t[j]
    #
    return h_t


"""
Note that this implementation assumes that the input arrays and kernels are NumPy arrays,
and it follows the equations for the standard GRU as described in the original paper by Cho et al. (2014).
The forward and backward passes are done separately for each timestep, with the hidden state bein
updated at each step. The output is stored in an output array and returned at the end.
"""




import numpy as np

def gru(input, f_kernel, f_r_kernel, f_bias, b_kernel, b_r_kernel, b_bias):
    """
    GRU (Gated Recurrent Unit) implementation.

    Parameters:
        input (np.array): Input array of shape (64,)
        f_kernel (np.array): Forward kernel of shape (64, 192)
        f_r_kernel (np.array): Forward recurrent kernel of shape (64, 192)
        f_bias (np.array): Forward bias of shape (2, 192)
        b_kernel (np.array): Backward kernel of shape (64, 192)
        b_r_kernel (np.array): Backward recurrent kernel of shape (64, 192)
        b_bias (np.array): Backward bias of shape (2, 192)

    Returns:
        output (np.array): Output array of shape (128,)
    """
    # Forward pass
    z = np.dot(input, f_kernel) + np.dot(input, f_r_kernel)
    z += f_bias[0]
    z = np.tanh(z)

    r = np.dot(input, f_kernel) + np.dot(input, f_r_kernel)
    r += f_bias[1]
    r = np.sigmoid(r)

    h = np.dot(input, f_kernel) + np.dot(r * input, f_r_kernel)
    h += f_bias[2]
    h = np.tanh(h)

    output_f = (1 - z) * h + z * input

    # Backward pass
    z = np.dot(input, b_kernel) + np.dot(input, b_r_kernel)
    z += b_bias[0]
    z = np.tanh(z)

    r = np.dot(input, b_kernel) + np.dot(input, b_r_kernel)
    r += b_bias[1]
    r = np.sigmoid(r)

    h = np.dot(input, b_kernel) + np.dot(r * input, b_r_kernel)
    h += b_bias[2]
    h = np.tanh(h)

    output_b = (1 - z) * h + z * input

    # Concatenate forward and backward outputs
    output = np.concatenate((output_f, output_b))

    return output










from data import *

output = [[0.0] * 128]

#print(output)
output = gru(
    gru_input,
    forward_kernel, forward_recurrent_kernel, forward_bias, 
    backward_kernel, backward_recurrent_kernel, backward_bias
)
print("#### gru called ####")
print("# OUTPUT #")
#print(output)
print("# EXPECTED #")
#print(gru_output_expected)
