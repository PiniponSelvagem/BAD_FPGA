#ifndef GRU_1_FORWARD_KERNEL_H
#define GRU_1_FORWARD_KERNEL_H
#include "../types.h"
#include "../size_bgru.h"

// Taken from: q_bidirectional_1_gru_forward_kernel.h
const gru_weigth_t g_1_forward_kernel[GRU1_KERNEL_SIZE] = {
    0.4854494333267212,
    -0.2713872194290161,
    -0.5504187941551208,
    -0.7145662903785706,
    -0.7964918613433838,
    -0.38654911518096924,
};

#endif // GRU_1_FORWARD_KERNEL_H