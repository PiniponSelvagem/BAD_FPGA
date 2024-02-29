#ifndef GRU_0_FORWARD_BIAS_H
#define GRU_0_FORWARD_BIAS_H
#include "../types.h"
#include "../size_bgru.h"

// Taken from: q_bidirectional_gru_forward_bias.h
const gru_weigth_t g_0_forward_bias[GRU_BIAS_SIZE] = {
    0.14834699034690857,
    0.10717877745628357,
    0.04259030893445015,
};

#endif // GRU_0_FORWARD_BIAS_H