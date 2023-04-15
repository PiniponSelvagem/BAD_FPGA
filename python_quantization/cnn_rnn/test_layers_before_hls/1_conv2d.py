from settings import C2D_1_ICHANNELS, C2D_1_IWIDTH, C2D_1_IHEIGHT, C2D_1_KSIZE, C2D_1_OWIDTH, C2D_1_OHEIGHT, C2D_OFFSET, IHEIGHT, IWIDTH
from data import conv2d_input_1, conv2d_kernel_1, conv2d_bias_1, conv2d_output_1


TOTAL_CHANNELS = 1 #C2D_1_ICHANNELS

def conv2D_1(input, weights, bias, output):
    for channel in range(TOTAL_CHANNELS-1):
        in_channel_offset = channel * C2D_1_IWIDTH * C2D_1_IHEIGHT
        out_channel_offset = channel * C2D_1_OWIDTH * C2D_1_OHEIGHT
        weight_offset = channel * C2D_1_KSIZE * C2D_1_KSIZE
        #
        for orow in range(1, int(C2D_1_OHEIGHT-C2D_OFFSET)-1):
            for ocol in range(int(C2D_OFFSET), int(C2D_1_OWIDTH-C2D_OFFSET)-1):
                acc = bias[channel]
                for krow in range(C2D_1_KSIZE-1):
                    for kcol in range(C2D_1_KSIZE-1):
                        weight_1d_loc = (krow * C2D_1_KSIZE + kcol) + weight_offset
                        image_1d_loc = int(((orow + krow - C2D_OFFSET) * C2D_1_IWIDTH + (ocol + kcol - C2D_OFFSET)) + in_channel_offset)
                        print("w ", weight_1d_loc, " - i ", image_1d_loc)
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
    return output   #



input = [0] * (C2D_1_OHEIGHT * C2D_1_OWIDTH * TOTAL_CHANNELS)
output = [0] * (C2D_1_OHEIGHT * C2D_1_OWIDTH * TOTAL_CHANNELS)


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


print(input)
print(output)
output = conv2D_1(conv2d_input_1, conv2d_kernel_1, conv2d_bias_1, conv2d_output_1)
print("conv2D_1 called")
print(output)