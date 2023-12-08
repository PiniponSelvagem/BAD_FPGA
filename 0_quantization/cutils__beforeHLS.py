import numpy as np
import os
import struct
from bitarray import bitarray


def createFolderIfNotExists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)



def saveArray_dim1(folder, fileName, array, arrayName, dataType):
    file_path = os.path.join(folder, '{}.h'.format(fileName))
    with open(file_path, 'w') as file:
        file.write('const {} {}[{}] = {{\n'.format(dataType, arrayName, array.size))
        for index, value in enumerate(array):
            file.write('    {},'.format(value))
            file.write('\n')
        file.write('};\n')


def saveArray_dim2(folder, fileName, array, arrayName, dataType):
    file_path = os.path.join(folder, '{}.h'.format(fileName))
    with open(file_path, 'w') as file:
        file.write('const {} {}[{}][{}] = '.format(dataType, arrayName, *array.shape))
        file.write('{\n')
        for i in range(array.shape[0]):
            file.write('    {\n')
            for j in range(array.shape[1]):
                file.write('        {},\n'.format(array[i, j]))
            file.write('    },\n')
        file.write('};\n')


def saveArray_dim3(folder, fileName, array, arrayName, dataType):
    file_path = os.path.join(folder, '{}.h'.format(fileName))
    with open(file_path, 'w') as file:
        file.write('const {} {}[{}][{}][{}] = '.format(dataType, arrayName, *array.shape))
        file.write('{\n')
        for i in range(array.shape[0]):
            file.write('    {\n')
            for j in range(array.shape[1]):
                file.write('        {\n')
                for k in range(array.shape[2]):
                    file.write('            {},\n'.format(array[i, j, k]))
                file.write('        },\n')
            file.write('    },\n')
        file.write('};\n')


def saveArray_dim4(folder, fileName, array, arrayName, dataType):
    file_path = os.path.join(folder, '{}.h'.format(fileName))
    with open(file_path, 'w') as file:
        file.write('const {} {}[{}][{}][{}][{}] = '.format(dataType, arrayName, *array.shape))
        file.write('{\n')
        for i in range(array.shape[0]):
            file.write('    {\n')
            for j in range(array.shape[1]):
                file.write('        {\n')
                for k in range(array.shape[2]):
                    file.write('            {\n')
                    for l in range(array.shape[3]):
                        file.write('                {},\n'.format(array[i, j, k, l]))
                    file.write('            },\n')
                file.write('        },\n')
            file.write('    },\n')
        file.write('};\n')



def packRow(packed_bits, array, dataType):
    if dataType["name"] == "ap_fixed":
        bits_total = dataType["bits_total"]
        bits_int = dataType["bits_int"]
        bits_dec = bits_total - bits_int
        #
        max_bits = bitarray('0' + '1' * (bits_total - 1))  # All bits except sign bit set to 1
        #
        # Convert the binary back to a float for max and min values
        max_value = int(max_bits.to01(), 2) / (2 ** bits_dec)
        min_value = -(1 << (bits_total - 1)) / (2 ** bits_dec)
        #
        #print(array)
        for value in array:
            #print(value)
            # Check for overflow and underflow
            if value > max_value:
                value = max_value
            elif value < min_value:
                value = min_value
            #
            # Scale the value to fit within the specified decimal_bits
            scaled_value = round(value * (2 ** bits_dec))
            #
            # Convert to one's complement binary
            if scaled_value >= 0:
                binary = bin(scaled_value)[2:].zfill(bits_total)
            else:
                # For negative values, calculate the one's complement
                complement = (1 << bits_total) + scaled_value
                binary = bin(complement)[2:].zfill(bits_total)
            #
            packed_bits.extend(binary)
        #
    else:
        # float
        binary_representation = struct.pack('f' * len(array), *array)
        bits = bitarray()
        bits.frombytes(binary_representation)
        packed_bits.extend(bits)
    return packed_bits

def packBits(packed_bits, row, dataType):
    row_data = [float(element) for element in row]
    packed_bits = packRow(packed_bits, row_data, dataType)
    return packed_bits


def saveArray_dim1_bin(folder, fileName, array, dataType):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_bits = bitarray()
    packed_bits = packBits(packed_bits, array, dataType)
    saveFileBin(file_path, packed_bits, dataType)

def saveArray_dim2_bin(folder, fileName, array, dataType):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_bits = bitarray()
    for row in array:
        packed_bits = packBits(packed_bits, row, dataType)
    saveFileBin(file_path, packed_bits, dataType)

def saveArray_dim3_bin(folder, fileName, array, dataType):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_bits = bitarray()
    for matrix in array:
        for row in matrix:
            packed_bits = packBits(packed_bits, row, dataType)
    saveFileBin(file_path, packed_bits, dataType)

def saveArray_dim4_bin(folder, fileName, array, dataType):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_bits = bitarray()
    for dim1 in array:
        for dim2 in dim1:
            for row in dim2:
                packed_bits = packBits(packed_bits, row, dataType)
    saveFileBin(file_path, packed_bits, dataType)


def saveFileBin(file_path, packed_bits, dataType):
    packed_data = b''
    if dataType["name"] == "ap_fixed":
        while len(packed_bits) % 8 != 0:
            # Pad the packed bits to a multiple of 8 if necessary
            packed_bits.append(False)
        #
        packed_data = packed_bits.tobytes()
    else:
        # float
        #packed_data = struct.pack('f' * len(array), *array)
        packed_data = packed_bits.tobytes()
    with open(file_path, 'wb') as file:
        file.write(packed_data)



def saveArray(folder, fileName, array, arrayName, dataType):
    size = len(array.shape)
    if size == 1:
        saveArray_dim1(folder, fileName, array, arrayName, "float")
        saveArray_dim1_bin(folder, fileName, array, dataType)
    elif size == 2:
        saveArray_dim2(folder, fileName, array, arrayName, "float")
        saveArray_dim2_bin(folder, fileName, array, dataType)
    elif size == 3:
        saveArray_dim3(folder, fileName, array, arrayName, "float")
        saveArray_dim3_bin(folder, fileName, array, dataType)
    elif size == 4:
        saveArray_dim4(folder, fileName, array, arrayName, "float")
        saveArray_dim4_bin(folder, fileName, array, dataType)
    else:
        print("ERROR saving array {}, with total dimensions {}.".format(arrayName, size))

