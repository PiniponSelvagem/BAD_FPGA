#ifndef GRU_1_BACKWARD_BIAS_RECURRENT_H
#define GRU_1_BACKWARD_BIAS_RECURRENT_H
#include "../types.h"
#include "../size_bgru.h"

// Taken from: q_bidirectional_1_gru_backward_bias_recurrent.h
const gru_weigth_t g_1_backward_bias_recurrent[GRU_BIAS_SIZE] = {
    -0.4412505626678467,
    -0.6330224871635437,
    0.06823571026325226,
};

#endif // GRU_1_BACKWARD_BIAS_RECURRENT_H