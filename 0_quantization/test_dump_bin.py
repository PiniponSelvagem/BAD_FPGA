import struct

#filename = "25_time_distributed_kernel_0"
#filename = "23_bidirectional_backward_gru_gru_cell_2_bias_0"
filename = "4_conv2d_1_kernel_0"

def print1D_f():
    with open('model_dump/'+filename+'.bin', 'rb') as file:
        binary_data = file.read()

        num_values = len(binary_data) // struct.calcsize('f')

        values = struct.unpack('f' * num_values, binary_data)

        for value in values:
            print(value)

def print2D_f(size0):
    with open('model_dump/'+filename+'.bin', 'rb') as file:
        binary_data = file.read()

        num_values = len(binary_data) // struct.calcsize('f')

        unpacked_data = struct.unpack('f' * num_values, binary_data)

        extracted_array = [unpacked_data[i:i+size0] for i in range(0, len(unpacked_data), size0)]

        for row in extracted_array:
            print(row)

def print3D_f(size1, size2):
    with open('model_dump/'+filename+'.bin', 'rb') as file:
        binary_data = file.read()

        num_values = len(binary_data) // struct.calcsize('f')

        unpacked_data = struct.unpack('f' * num_values, binary_data)

        extracted_array = []
        num_cols = size1
        num_rows = size2

        for i in range(0, len(unpacked_data), num_cols):
            matrix_values = unpacked_data[i:i+num_cols]
            extracted_array.append([matrix_values[j:j+num_rows] for j in range(0, len(matrix_values), num_rows)])

        for matrix in extracted_array:
            print(matrix)

def print4D_f(size0, size1, size2):
    with open('model_dump/'+filename+'.bin', 'rb') as file:
        binary_data = file.read()

        num_values = len(binary_data) // struct.calcsize('f')

        unpacked_data = struct.unpack('f' * num_values, binary_data)

        extracted_array = []
        dim1_len = size0
        dim2_len = size1
        dim3_len = size2

        for i in range(0, len(unpacked_data), dim3_len):
            dim3_values = unpacked_data[i:i+dim3_len]
            dim2_values = [dim3_values[j:j+dim3_len] for j in range(0, len(dim3_values), dim3_len)]
            dim1_values = [dim2_values[k:k+dim2_len] for k in range(0, len(dim2_values), dim2_len)]
            extracted_array.append(dim1_values)

        for dim1 in extracted_array:
            for dim2 in dim1:
                for dim3 in dim2:
                    print(dim3)



#print1D_f()
#print2D_f(64)
#print3D_f(3, 64)
print4D_f(64, 64, 3)