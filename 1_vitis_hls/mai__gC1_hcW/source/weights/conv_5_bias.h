#ifndef CONV_5_BIAS_H
#define CONV_5_BIAS_H
#include "../types.h"

// Taken from: q_conv2d_batchnorm_5_bias_hls.h
const bias_t bias_5[CHANNELS] = {
    0.0,
    512.0,
    0.0,
    -512.0,
    -1792.0,
    512.0,
    512.0,
    -1024.0,
    -256.0,
    -2048.0,
    1536.0,
    -512.0,
    -1536.0,
    0.0,
    -768.0,
    1536.0,
    -1536.0,
    256.0,
    -1024.0,
    -2048.0,
    -512.0,
    0.0,
    -512.0,
    1792.0,
    0.0,
    512.0,
    -768.0,
    256.0,
    -1024.0,
    -2048.0,
    1024.0,
    1024.0,
    -3072.0,
    -2048.0,
    256.0,
    -768.0,
    -512.0,
    -1536.0,
    1792.0,
    1024.0,
    1536.0,
    -2048.0,
    -768.0,
    -1792.0,
    -512.0,
    -2560.0,
    -1536.0,
    1024.0,
    -2560.0,
    -1536.0,
    -2560.0,
    1792.0,
    -2048.0,
    256.0,
    -1792.0,
    -512.0,
    -512.0,
    -512.0,
    1024.0,
    0.0,
    -1536.0,
    512.0,
    -1280.0,
    -1536.0,
};

#endif // CONV_5_BIAS_H