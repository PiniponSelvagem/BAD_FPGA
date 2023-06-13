from settings import C2D_1_ICHANNELS, C2D_1_IWIDTH, C2D_1_IHEIGHT, C2D_1_KSIZE, C2D_1_OWIDTH, C2D_1_OHEIGHT, C2D_OFFSET, IHEIGHT, IWIDTH
from data import conv2d_input_1, conv2d_kernel_1, conv2d_kernel_1_b, conv2d_kernel_1_c, conv2d_bias_1, conv2d_output_1


TOTAL_CHANNELS = 1 #C2D_1_ICHANNELS

def conv2D_1(input, weights, bias):
    """
    NOT WORKING
    """
    for channel in range(TOTAL_CHANNELS):
        in_channel_offset = channel * C2D_1_IWIDTH * C2D_1_IHEIGHT
        out_channel_offset = channel * C2D_1_OWIDTH * C2D_1_OHEIGHT
        weight_offset = channel * C2D_1_KSIZE * C2D_1_KSIZE
        #
        for orow in range(1, int(C2D_1_OHEIGHT-C2D_OFFSET)):
            for ocol in range(int(C2D_OFFSET), int(C2D_1_OWIDTH-C2D_OFFSET)):
                acc = bias[channel]
                for krow in range(C2D_1_KSIZE):
                    for kcol in range(C2D_1_KSIZE):
                        weight_1d_loc = (krow * C2D_1_KSIZE + kcol) + weight_offset
                        image_1d_loc = int(((orow + krow - C2D_OFFSET) * C2D_1_IWIDTH + (ocol + kcol - C2D_OFFSET)) + in_channel_offset)
                        #print("w ", weight_1d_loc, " - i ", image_1d_loc)
                        acc += weights[weight_1d_loc] * input[image_1d_loc]
                #
                # ReLu
                if (acc > 255):
                    acc_sat = 255
                elif (acc < 0):
                    acc_sat = 0
                else:
                    acc_sat = acc
                #
                output[(orow * C2D_1_OWIDTH + ocol) + out_channel_offset] = acc_sat
    #
    return output



import numpy as np
def conv2D(image, kernel, bias):
    # Convert inputs to NumPy arrays
    image = np.array(image)
    kernel = np.array(kernel)
    bias = np.array(bias)
    
    # Get image and kernel dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'constant')

    # Create output array
    output = np.zeros((image_height, image_width))
    
    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            output[i][j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)
            
            # Add bias term
            output[i][j] += bias
    
    return output


input = [0] * (C2D_1_OHEIGHT * C2D_1_OWIDTH * TOTAL_CHANNELS)
output = [0] * (3 * 42 * 1)


"""
for channel in range(TOTAL_CHANNELS):
    channel_offset = channel * C2D_1_IWIDTH * C2D_1_IHEIGHT
    for orow in range(C2D_1_IHEIGHT-1):
        for ocol in range(C2D_1_IWIDTH-1):
            index = (orow * C2D_1_IWIDTH + ocol) + channel_offset
            if (orow < C2D_OFFSET) or (orow >= C2D_OFFSET+IHEIGHT) or (ocol < C2D_OFFSET) or (ocol >= C2D_OFFSET+IWIDTH):
                input[index] = 0
            else:
                value = conv2d_input_1[int((orow-C2D_OFFSET) * IWIDTH + (ocol-C2D_OFFSET))]
                input[index] = value
"""



input_debug = [
    [
        1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
    ]
]
weights_debug = [ 
    [ 0, 0, 0 ],
    [ 0, 1, 0 ],
    [ 0, 0, 0 ],
]
bias_debug = 1

#print(output)
#output = conv2D_1(output, conv2d_kernel_1, conv2d_bias_1)
#output = conv2D(conv2d_input_1, conv2d_kernel_1_c, conv2d_bias_1)
output = conv2D(conv2d_input_1, conv2d_kernel_1_b, conv2d_bias_1)
#output = conv2D(input_debug, weights_debug, bias_debug)
print("DONE")
print(output)
print("\nEXPECTED:")
print(conv2d_output_1)
