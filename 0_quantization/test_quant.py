def float_to_bits(num, num_bits):
    # Get the integer part of the float
    integer_part = int(num)

    # Convert the integer part to its binary representation
    integer_bits = bin(abs(integer_part))[2:]

    # Determine the sign bit
    sign_bit = '1' if integer_part < 0 else '0'

    # Add leading zeros if necessary
    integer_bits = integer_bits.zfill(num_bits - 1)  # Leave one bit for the sign

    # Apply two's complement if the number is negative
    if integer_part < 0:
        # Flip the bits using XOR operation
        integer_bits = ''.join('1' if bit == '0' else '0' for bit in integer_bits)

        # Add 1 to the flipped bits
        integer_bits = bin(int(integer_bits, 2) + 1)[2:].zfill(len(integer_bits))

    # Combine the sign bit and integer bits
    binary_representation = sign_bit + integer_bits

    return binary_representation

# Example usage
float_num = -3.75
bits = float_to_bits(float_num, 6)  # Using 6 bits for the integer part
##print(bits)



def bits_to_float(bits):
    # Extract the sign bit and integer bits
    sign_bit = bits[0]
    integer_bits = bits[1:]

    # Convert the integer bits back to an integer
    integer_part = int(integer_bits, 2)

    # Apply two's complement if the sign bit is '1'
    if sign_bit == '1':
        integer_part -= 2 ** (len(bits) - 1)

    return integer_part

#
integer_part = bits_to_float(bits)
##print(integer_part)






from bitarray import bitarray
"""
bits_int = 4
bits_dec = 9
value = -3.125
print("value "+str(value))


packed_bits = bitarray()
integer_part = int(value)
decimal_part = abs(int(round((value - integer_part) * (1 << bits_dec))))

integer_part &= (1 << bits_int) - 1
decimal_part &= (1 << bits_dec) - 1

print("int " + format(integer_part, '0{}b'.format(bits_int)))
print("dec " + format(decimal_part, '0{}b'.format(bits_dec)))

packed_bits.extend(format(integer_part, '0{}b'.format(bits_int)))
packed_bits.extend(format(decimal_part, '0{}b'.format(bits_dec)))

print(packed_bits)
"""



from bitarray import bitarray
import struct

bits_int = 1 #7
bits_dec = 4-bits_int #16

file_path = 'model_quantized_weights/conv2d_kernel.bin'

# Read the packed bits from the binary file
with open(file_path, 'rb') as file:
    packed_bytes = file.read()

# Convert the bytes to a bitarray
packed_bits = bitarray()
packed_bits.frombytes(packed_bytes)




print("#################")
print("#################")


"""
Bit representation: 0100 (0.5)
Bit representation: 1100 (-0.5)
Bit representation: 0010 (0.25)
Bit representation: 1110 (-0.25)
Bit representation: 0111 (0.875)
Bit representation: 1001 (-0.875)
Bit representation: 0001 (0.125)
"""

# Define the values
#values = [0.5, -0.5, 0.25, -0.25, 0.875, -0.875, 0.125]
values = [
    -0.109375,
    -0.09375,
    0.0625,
    0.015625,
    -0.078125,
    0.0,
    -0.0625,
    0.03125,
    0.046875,

    -0.01171875,
    -0.001953125,
    -0.001953125,
    -0.001953125,
    -0.013671875,
    -0.013671875,
    -0.005859375,
    -0.01171875,
    -0.013671875,

    10,
    0.5,
    -10,
    0.5,
    0.000001,
    0.5,
    -0.000001
]

# Define the number of bits for integer and decimal parts
total_bits = 8
integer_bits = 1
decimal_bits = total_bits - integer_bits

# Initialize packed_bits as an empty bitarray
max_bits = bitarray('0' + '1' * (total_bits - 1))  # All bits except sign bit set to 1
min_bits = bitarray('1' + '0' * (total_bits - 1))  # Sign bit set to 1, rest set to 0

# Convert the binary back to a float for max and min values
max_value = int(max_bits.to01(), 2) / (2 ** decimal_bits)
min_value = -(1 << (total_bits - 1)) / (2 ** decimal_bits)

# Print the max and min values
print("Max value:", max_value)
print("Min value:", min_value)
print("Max value in bits:", max_bits)
print("Min value in bits:", min_bits)

# Function to convert a float to one's complement binary
def float_to_ones_complement_binary(value, integer_bits, decimal_bits):
    # Check for overflow and underflow
    if value > max_value:
        value = max_value
    elif value < min_value:
        value = min_value
    
    # Scale the value to fit within the specified decimal_bits
    scaled_value = round(value * (2 ** decimal_bits))
    
    # Convert to one's complement binary
    if scaled_value >= 0:
        binary = bin(scaled_value)[2:].zfill(total_bits)
    else:
        # For negative values, calculate the one's complement
        complement = (1 << total_bits) + scaled_value
        binary = bin(complement)[2:].zfill(total_bits)
    
    return binary

# Print the bit representations
packed_bits = bitarray()
for value in values:
    binary = float_to_ones_complement_binary(value, integer_bits, decimal_bits)
    packed_bits.extend(binary)
    
    # Convert the binary back to a float
    if binary[0] == '0':
        converted_value = int(binary, 2) / (2 ** decimal_bits)
    else:
        # Handle one's complement for negative values
        complement = int(binary, 2)
        converted_value = -(2 ** total_bits - complement) / (2 ** decimal_bits)
    
    # Use string formatting to specify fixed widths for both values
    print("Bit representation: {:>8} ({:>9.5f}), Converted value: {:>9.5f}".format(binary, value, converted_value))

print(packed_bits)