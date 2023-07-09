import numpy as np
import os
import struct


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




def saveArray_dim1_bin(folder, fileName, array, dataType):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_data = struct.pack('f' * len(array), *array)
    with open(file_path, 'wb') as file:
        file.write(packed_data)

def saveArray_dim2_bin(folder, fileName, array, dataType):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_data = b''
    for row in array:
        row_float = [float(element) for element in row]
        packed_row = struct.pack('f' * len(row_float), *row_float)
        packed_data += packed_row
    with open(file_path, 'wb') as file:
        file.write(packed_data)

def saveArray_dim3_bin(folder, fileName, array, dataType):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_data = b''
    for matrix in array:
        for row in matrix:
            row_float = [float(element) for element in row]
            packed_row = struct.pack('f' * len(row_float), *row_float)
            packed_data += packed_row
    with open(file_path, 'wb') as file:
        file.write(packed_data)

def saveArray_dim4_bin(folder, fileName, array, dataType):
    file_path = os.path.join(folder, '{}.bin'.format(fileName))
    packed_data = b''
    for dim1 in array:
        for dim2 in dim1:
            for dim3 in dim2:
                row_float = [float(element) for element in dim3]
                packed_row = struct.pack('f' * len(row_float), *row_float)
                packed_data += packed_row
    with open(file_path, 'wb') as file:
        file.write(packed_data)


def saveArray(folder, fileName, array, arrayName, dataType):
    size = len(array.shape)
    if size == 1:
        saveArray_dim1(folder, fileName, array, arrayName, dataType)
        saveArray_dim1_bin(folder, fileName, array, dataType)
    elif size == 2:
        saveArray_dim2(folder, fileName, array, arrayName, dataType)
        saveArray_dim2_bin(folder, fileName, array, dataType)
    elif size == 3:
        saveArray_dim3(folder, fileName, array, arrayName, dataType)
        saveArray_dim3_bin(folder, fileName, array, dataType)
    elif size == 4:
        saveArray_dim4(folder, fileName, array, arrayName, dataType)
        saveArray_dim4_bin(folder, fileName, array, dataType)
    else:
        print("ERROR saving array {}, with total dimensions {}.".format(arrayName, size))

