#ifndef CONV_2_KERNEL_SCALE_H
#define CONV_2_KERNEL_SCALE_H
#include "../types.h"
#include "../size_conv3D.h"

// Taken from: q_conv2d_batchnorm_2_kernel_scale_hls_packet.h
const weigth_t kernel_2_scale[CHANNELS/PACKET_CNN] = {
    0x6766676656776666,
    0x5666666776766666,
    0x7667766766766666,
    0x7667666666667677,
};

#endif // CONV_2_KERNEL_SCALE_H