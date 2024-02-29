#ifndef CONV_0_KERNEL_SCALE_H
#define CONV_0_KERNEL_SCALE_H
#include "../types.h"
#include "../size_conv3D.h"

// Taken from: q_conv2d_batchnorm_kernel_scale_hls_packet.h
const weigth_t kernel_0_scale[CHANNELS/PACKET_CNN] = {
    0x1150110511512402,
    0x11000140112131,
    0x1101012111011221,
    0x3211112111231210,
};

#endif // CONV_0_KERNEL_SCALE_H