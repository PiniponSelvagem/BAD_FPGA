#ifndef GRU_1_FORWARD_BIAS_H
#define GRU_1_FORWARD_BIAS_H
#include "../types.h"
#include "../size_bgru.h"

// Taken from: q_bidirectional_1_gru_forward_bias.h
const gru_weigth_t g_1_forward_bias[GRU_BIAS_SIZE] = {
    -0.40473973751068115,
    0.4012516736984253,
    -0.1887999176979065,
};

#endif // GRU_1_FORWARD_BIAS_H