#ifndef GRU_1_BACKWARD_KERNEL_RECURRENT_H
#define GRU_1_BACKWARD_KERNEL_RECURRENT_H
#include "../types.h"
#include "../size_bgru.h"

// Taken from: q_bidirectional_1_gru_backward_recurrent_kernel.h
const gru_weigth_t g_1_backward_kernel_recurrent[GRU_RKERNEL_SIZE] = {
    0.06656814366579056,
    -0.10812682658433914,
    -0.14362101256847382,
};

#endif // GRU_1_BACKWARD_KERNEL_RECURRENT_H