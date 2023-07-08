import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dense_layer(input, weights, bias):
    output = np.dot(input, weights) + bias
    return sigmoid(output)

def time_distributed(input, weights, bias):
    batch_size, sequence_length, input_dim = input.shape
    output_dim = weights.shape[1]

    output = np.zeros((batch_size, sequence_length, output_dim))
    for b in range(batch_size):
        for t in range(sequence_length):
            input_slice = input[b, t, :]
            output_slice = dense_layer(input_slice, weights)
            output[b, t, :] = output_slice

    return output

# Example usage
batch_size = 2
sequence_length = 3
input_dim = 4
output_dim = 5

# Create input tensor
input = np.random.rand(batch_size, sequence_length, input_dim)
print(input)

# Create weights tensor for dense layer
weights = np.random.rand(input_dim, output_dim)
print(weights)

# Apply TimeDistributed layer
output = time_distributed(input, weights)

# Print output tensor
for b in range(batch_size):
    for t in range(sequence_length):
        print(f"Output for batch {b}, time step {t}:")
        print(output[b, t, :])
        print()
