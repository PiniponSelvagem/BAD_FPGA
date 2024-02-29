#ifndef CONV_1_KERNEL_SCALE_H
#define CONV_1_KERNEL_SCALE_H
#include "../types.h"
#include "../size_conv3D.h"

// Taken from: q_conv2d_batchnorm_1_kernel_scale_hls_packet.h
const weigth_t kernel_1_scale[CHANNELS/PACKET_CNN] = {
    0x4445455544554444,
    0x4544444464444554,
    0x3445554545554554,
    0x5544554445444444,
};

#endif // CONV_1_KERNEL_SCALE_H