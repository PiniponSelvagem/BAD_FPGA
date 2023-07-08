import numpy as np

def activation(x):
    return x

input_size = 10
hidden_size = 4

# Randomly initialize weights and biases
weights = np.random.randn(input_size, hidden_size)
biases = np.random.randn(hidden_size)

# Input data
input_data = np.random.randn(input_size)

# Compute the output of the dense layer
output = activation(np.dot(input_data, weights) + biases)

print(output)
