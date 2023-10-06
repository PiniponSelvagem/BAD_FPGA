# READS ONLY BIN FILE WITH FLOATS
import struct

def read_floats_from_binary_file(file_path, num_floats):
    floats = []
    with open(file_path, "rb") as file:
        for _ in range(num_floats):
            data = file.read(4)  # Assuming each float is 4 bytes (32 bits)
            if not data:
                break  # Reached the end of the file
            float_value = struct.unpack("f", data)[0]
            floats.append(float_value)
    return floats

# Usage example
file_path = "model_quantized_weights/conv2d_kernel.bin"
num_floats_to_read = 18
floats_list = read_floats_from_binary_file(file_path, num_floats_to_read)
print(floats_list)