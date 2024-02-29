#ifndef CONV_4_KERNEL_SCALE_H
#define CONV_4_KERNEL_SCALE_H
#include "../types.h"
#include "../size_conv3D.h"

// Taken from: q_conv2d_batchnorm_4_kernel_scale_hls_packet.h
const weigth_t kernel_4_scale[CHANNELS/PACKET_CNN] = {
    0x7777676777777676,
    0x7766767667777777,
    0x7776677667777776,
    0x6677677776766766,
};

#endif // CONV_4_KERNEL_SCALE_H