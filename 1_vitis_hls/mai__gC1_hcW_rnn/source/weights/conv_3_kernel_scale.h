#ifndef CONV_3_KERNEL_SCALE_H
#define CONV_3_KERNEL_SCALE_H
#include "../types.h"
#include "../size_conv3D.h"

// Taken from: q_conv2d_batchnorm_3_kernel_scale_hls_packet.h
const weigth_t kernel_3_scale[CHANNELS/PACKET_CNN] = {
    0x6667767666667666,
    0x6766666776777666,
    0x6767776667676667,
    0x6776676776667667,
};

#endif // CONV_3_KERNEL_SCALE_H