import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dense_layer(input, weights, bias):
    output = np.dot(input, weights) + bias
    return sigmoid(output)

def time_distributed(input, weights, bias):
    batch_size, sequence_length, _, input_dim = input.shape
    _, output_dim = weights.shape

    output = np.zeros((batch_size, sequence_length, input_dim, output_dim))
    print(output.shape)
    for b in range(batch_size):
        for t in range(sequence_length):
            input_slice = input[b, t]
            output_slice = dense_layer(input_slice, weights, bias)
            output[b, t] = output_slice

    return output

# Example usage
batch_size = 2
sequence_length = 3
input_dim = 4

# Set your own input, kernel, and bias
input = np.array([
    [
        [[1, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
    ],
    [
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
    ]
])

kernel = np.array([
    [0, 0],
    [0, 0],
])

bias = np.array([0, 0])

# Apply TimeDistributed layer
output_tensor = time_distributed(input, kernel, bias)

# Print output tensor
for b in range(batch_size):
    for t in range(sequence_length):
        print(f"Output for batch {b}, time step {t}:")
        print(output_tensor[b, t, :])
        print()

