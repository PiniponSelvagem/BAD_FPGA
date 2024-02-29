#ifndef CONV_5_KERNEL_SCALE_H
#define CONV_5_KERNEL_SCALE_H
#include "../types.h"
#include "../size_conv3D.h"

// Taken from: q_conv2d_batchnorm_5_kernel_scale_hls_packet.h
const weigth_t kernel_5_scale[CHANNELS/PACKET_CNN] = {
    0x6667777666767666,
    0x7776666766777666,
    0x7776667766766677,
    0x7676777676676777,
};

#endif // CONV_5_KERNEL_SCALE_H