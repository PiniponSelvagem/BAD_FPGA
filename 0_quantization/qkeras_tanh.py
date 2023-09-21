import tensorflow as tf
from qkeras import QActivation
import matplotlib.pyplot as plt

bits = 4
quantized_tanh = QActivation("quantized_tanh(%d)" % bits)

# Define the range and step for input values
start_range = -2.0
end_range = 2.0
step = 0.001

# Generate the input values using the specified range and step
input_values = tf.constant(
    [start_range + i * step for i in range(int((end_range - start_range) / step) + 1)],
    dtype=tf.float32,
)

output_values_quantized_tanh = quantized_tanh(input_values)
output_values_tf_tanh = tf.math.tanh(input_values)


def custom_tanh(x):
    x = x + 0.0625      # add offset
    step_size = 0.125
    if x <= -1:
        return -1.0
    elif x >= 0.875:
        return 0.875
    else:
        step = int((x + 1) / step_size)
        return (step * step_size) - 1.0


output_values_custom_tanh = [custom_tanh(x) for x in input_values]


# Initialize a variable to keep track of the previous comparison result
previous_match = True

# Compare and print values
for x, custom_result, quantized_result in zip(input_values, output_values_custom_tanh, output_values_quantized_tanh):
    if custom_result != quantized_result:
        if not previous_match:
            print(f"x = {x:.3f}, custom_tanh(x) = {custom_result:.3f}, quantized_tanh(x) = {quantized_result:.3f}")
        previous_match = False
    else:
        previous_match = True



plt.figure(figsize=(8, 6))
plt.plot(input_values, output_values_quantized_tanh, label="Quantized Tanh (4-bit)")
plt.plot(input_values, output_values_tf_tanh, label="TensorFlow Tanh")
plt.plot(input_values, output_values_custom_tanh, label="Custom Tanh")
plt.xlabel("Input Value")
plt.ylabel("Activation Value")
plt.legend()
plt.title("Quantized Tanh vs. TensorFlow Tanh Activation")
plt.grid(True)
plt.savefig("quantized_tanh_plot.png")
plt.close()



"""
# Print the values of the three outputs on the same line with labels
for x, tf_val, qk_val, custom_val in zip(input_values, output_values_tf_tanh, output_values_quantized_tanh, output_values_custom_tanh):
    print(f"Input: {x:.8f} | TF: {tf_val:.8f} | QK: {qk_val:.8f} | LT: {custom_val:.8f}")
"""










"""
def custom_tanh(x):
    # Define the lookup table
    lookup_table = [
        -1.000000,
        -0.875000,
        -0.750000,
        -0.625000,
        -0.500000,
        -0.375000,
        -0.250000,
        -0.125000,
        0.000000,
        0.125000,
        0.250000,
        0.375000,
        0.500000,
        0.625000,
        0.750000,
        0.875000,
    ]
    
    # Ensure that the input x is within the range [-1, 1]
    x = max(-1.0, min(1.0, x))
    
    # Calculate the index for the lookup table
    index = int((x + 1.0) * 8)

    if index > (len(lookup_table)-1):
        index = len(lookup_table)-1
    elif index < 0:
        index = 0
    
    # Perform the lookup
    approximated_tanh = lookup_table[index]
    
    return approximated_tanh

# Test the custom tanh function
input_x = [-1.0, -0.5, 0.0, 0.5, 1.0]
for x in input_x:
    approx_tanh = custom_tanh(x)
    qkeras_tanh_value = quantized_tanh(x)
    print(f"Input: {x}, Approximated tanh: {approx_tanh}, QKeras tanh: {qkeras_tanh_value}")
"""











"""
# Initialize previous_output with an out-of-range value to ensure the first entry is always saved
previous_output = -999.99

# Create a lookup table for quantized tanh values using regular Python lists
lookup_table = {}

# Populate the lookup table with input values and their corresponding quantized tanh outputs
for i in range(len(input_values)):
    input_value = float(input_values[i])
    output_value = quantized_tanh(tf.constant(input_value, dtype=tf.float32)).numpy()
    if output_value != previous_output:
        lookup_table[input_value] = output_value
        previous_output = output_value

# Function to find the closest value in the lookup table and return its quantized output
def find_closest_quantized_tanh(input_value):
    closest_input = min(lookup_table.keys(), key=lambda x: abs(x - input_value))
    return lookup_table[closest_input]

# Print the unique quantized output values from the lookup table
unique_quantized_outputs = set(lookup_table.values())
print("Unique Quantized Output Values:")
for quantized_output in unique_quantized_outputs:
    print(f"Quantized Output: {quantized_output:f}")
"""

