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



def packData(packed_bits, array, dataType, saveBinAsInteger=False, binPositiveOnly=False):
    # saveBinAsInteger should be True when Kernel is merged with its Scale making the values of the resulted kernel Integers.
    #       knowing that, the values should be saved without the fracction pre-processing
    # binPositiveOnly is only evaluated if saveBinAsInteger is set to False
    if dataType["name"] == "ap_fixed":
        bits_total = dataType["bits_total"]
        bits_int = dataType["bits_int"]
        bits_dec = bits_total - bits_int
        #
        if saveBinAsInteger:
            max_bits = bitarray('0' + '1' * (bits_total - 1))
            #
            max_value = int(max_bits.to01(), 2)
            min_value = -(1 << (bits_total - 1))
        else:
            if binPositiveOnly:
                max_bits = bitarray('1' * (bits_total))
                #
                max_value = int(max_bits.to01(), 2)
                min_value = 0
            else:
                max_bits = bitarray('0' + '1' * (bits_total - 1))  # All bits except sign bit set to 1
                #
                # Convert the binary back to a float for max and min values
                max_value = int(max_bits.to01(), 2) / (2 ** bits_dec)
                min_value = -(1 << (bits_total - 1)) / (2 ** bits_dec)
        #
        #
        # NOTE: The isOddElement check was only tested with 4 total bits, and it is hardcoded to only work in those situations.
        #       It is used to save the bytes in litle-endian, making sure the lower part is placed in the lower part of the byte.
        #       It should not be necessary to use this technique in 8, 16, 32, etc bits.
        isOddElement = False
        prevBinary = "0000"
        for value in array:
            # Check for overflow and underflow
            if value > max_value:
                value = max_value
            elif value < min_value:
                value = min_value
            #
            if saveBinAsInteger:
                scaled_value = int(value)
            else:
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
            #print(value, ",", scaled_value, " --- ", binary)
            #
            if bits_total == 4:
                if isOddElement:
                    packed_bits.extend(binary)
                    packed_bits.extend(prevBinary)
                    isOddElement = False
                else:
                    prevBinary = binary
                    isOddElement = True
            else:
                packed_bits.extend(binary)
        if bits_total == 4: # closing in the case of 4 bits_total
            if isOddElement:
                packed_bits.extend(prevBinary)
                """
            else:
                # This is wrong, do not use it.
                # Was trying to make a case for situations that the array was not even.
                packed_bits.extend(binary)
                packed_bits.extend("1110")  # close the leftover byte
                """
    elif dataType["name"] == "ap_int16":
        # int16
        int_array = [int(value) for value in array]
        binary_representation = struct.pack('h' * len(int_array), *int_array)
        bits = bitarray()
        bits.frombytes(binary_representation)
        packed_bits.extend(bits)
    else:
        # float
        binary_representation = struct.pack('f' * len(array), *array)
        bits = bitarray()
        bits.frombytes(binary_representation)
        packed_bits.extend(bits)
    return packed_bits

def packBits(packed_bits, array, dataType, saveBinAsInteger, binPositiveOnly):
    data = array.flatten()
    packed_bits = packData(packed_bits, data, dataType, saveBinAsInteger, binPositiveOnly)
    return packed_bits


def saveArray_bin(folder, fileName, array, dataType, saveBinAsInteger, binPositiveOnly):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_bits = bitarray()
    packed_bits = packBits(packed_bits, array, dataType, saveBinAsInteger, binPositiveOnly)
    #print(packed_bits)
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



def saveArray(folder, fileName, array, arrayName, dataType, saveBinAsInteger=False, binPositiveOnly=False):
    size = len(array.shape)
    if size == 1:
        saveArray_dim1(folder, fileName, array, arrayName, "float")
        saveArray_bin(folder, fileName, array, dataType, saveBinAsInteger, binPositiveOnly)
    elif size == 2:
        saveArray_dim2(folder, fileName, array, arrayName, "float")
        saveArray_bin(folder, fileName, array, dataType, saveBinAsInteger, binPositiveOnly)
    elif size == 3:
        saveArray_dim3(folder, fileName, array, arrayName, "float")
        saveArray_bin(folder, fileName, array, dataType, saveBinAsInteger, binPositiveOnly)
    elif size == 4:
        saveArray_dim4(folder, fileName, array, arrayName, "float")
        saveArray_bin(folder, fileName, array, dataType, saveBinAsInteger, binPositiveOnly)
    else:
        print("ERROR saving array {}, with total dimensions {}.".format(arrayName, size))

