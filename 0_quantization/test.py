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

bits_int = 8 #7
bits_dec = 32-bits_int #16

file_path = 'model_dump/1_conv2d_bias_0.bin'

# Read the packed bits from the binary file
with open(file_path, 'rb') as file:
    packed_bytes = file.read()

# Convert the bytes to a bitarray
packed_bits = bitarray()
packed_bits.frombytes(packed_bytes)

# Calculate the number of packed values in the file
num_packed_values = len(packed_bits) // (bits_int + bits_dec)

# Iterate over the packed values and print the combined float value
for i in range(num_packed_values):
    start_index = i * (bits_int + bits_dec)
    end_index = start_index + bits_int
    integer_part = int(packed_bits[start_index:end_index].to01(), 2)

    start_index = end_index
    end_index += bits_dec
    decimal_part = int(packed_bits[start_index:end_index].to01(), 2)

    combined_value = integer_part + decimal_part / (2 ** bits_dec)

    # Pack the combined value as a 32-bit signed float (4 bytes)
    packed_value = struct.pack('f', combined_value)

    # Unpack the packed value as a float
    unpacked_value = struct.unpack('f', packed_value)[0]

    print("Value {}: int {}, dec {}, float {}".format(i + 1, format(integer_part, '0{}b'.format(bits_int)),
                                                                     format(decimal_part, '0{}b'.format(bits_dec)),
                                                                     unpacked_value))
    





print("#################")
print("#################")


#array = [35.51, 2.5, -0.5, 0.025, -0.025, 0.05, -0.05, -3.125, 0.02698972076177597]
array = [-0.005151249002665281]
bits_int = 7
bits_dec = 16-bits_int
packed_bits = bitarray()
for value in array:
    is_neg = value < 0
    integer_part = abs(int(value))
    decimal_part = abs(int(round((value - integer_part) * (1 << bits_dec))))
    #
    if is_neg:  # Convert to one's complement
        integer_part = ~integer_part & ((1 << bits_int) - 1)
        integer_part |= 1 << (bits_int - 1)
        #
        decimal_part = ~decimal_part
        decimal_part = decimal_part+1
    #
    integer_part &= (1 << bits_int) - 1
    decimal_part &= (1 << bits_dec) - 1
    #
    packed_bits.extend(format(integer_part, '0{}b'.format(bits_int)))
    packed_bits.extend(format(decimal_part, '0{}b'.format(bits_dec)))
#
while len(packed_bits) % 8 != 0:
    # Pad the packed bits to a multiple of 8 if necessary
    packed_bits.append(False)
#
print(str(packed_bits) + " - value: " + str(value))




print("$$$$$$$")

def float_to_fixed_point(value, integer_bits, fractional_bits):
    # Determine the range and resolution of fixed-point representation
    int_range = 2 ** (integer_bits - 1) - 1  # Integer range (-64 to 63 in this case)
    frac_resolution = 1 / (2 ** fractional_bits)  # Fractional resolution (1/512 in this case)

    # Scale the value based on the fractional range
    scaled_value = round(value * (2 ** fractional_bits))

    # Perform range checks and handle overflow/underflow
    fixed_point_value = min(max(scaled_value, -int_range), int_range)

    return fixed_point_value


def fixed_point_to_float(value, integer_bits, fractional_bits):
    # Determine the range and resolution of fixed-point representation
    int_range = 2 ** (integer_bits - 1) - 1  # Integer range (-64 to 63 in this case)
    frac_resolution = 1 / (2 ** fractional_bits)  # Fractional resolution (1/512 in this case)

    # Scale the value based on the fractional range
    scaled_value = value / (2 ** fractional_bits)

    return scaled_value


v = float_to_fixed_point(-0.005151249002665281, 7, 9)
v = fixed_point_to_float(v, 7, 9)
print(v)

v = float_to_fixed_point(0.2148435115814209, 7, 9)
v = fixed_point_to_float(v, 7, 9)
print(v)