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



def packRow(array, dataType):
    if dataType["name"] == "ap_fixed":
        bits_int = dataType["bits_int"]
        bits_dec = dataType["bits_total"] - bits_int
        #
        packed_bits = bitarray()
        #print(array)
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
            #print(packed_bits)
            packed_bits.extend(format(decimal_part, '0{}b'.format(bits_dec)))
            #print(packed_bits)
            #print("value: " + str(value))
        #
        while len(packed_bits) % 8 != 0:
            # Pad the packed bits to a multiple of 8 if necessary
            packed_bits.append(False)
        #
        pack = packed_bits.tobytes()
    else:
        # float
        pack = struct.pack('f' * len(array), *array)
    return pack

def packData(row, dataType):
    row_data = [float(element) for element in row]
    packed_row = packRow(row_data, dataType)
    return packed_row


def saveArray_dim1_bin(folder, fileName, array, dataType):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_data = packRow(array, dataType)
    with open(file_path, 'wb') as file:
        file.write(packed_data)

def saveArray_dim2_bin(folder, fileName, array, dataType):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_data = b''
    for row in array:
        packed_data += packData(row, dataType)
    with open(file_path, 'wb') as file:
        file.write(packed_data)

def saveArray_dim3_bin(folder, fileName, array, dataType):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_data = b''
    for matrix in array:
        for row in matrix:
            packed_data += packData(row, dataType)
    with open(file_path, 'wb') as file:
        file.write(packed_data)

def saveArray_dim4_bin(folder, fileName, array, dataType):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_data = b''
    for dim1 in array:
        for dim2 in dim1:
            for row in dim2:
                packed_data += packData(row, dataType)
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

