#ifndef GRU_0_FORWARD_KERNEL_RECURRENT_H
#define GRU_0_FORWARD_KERNEL_RECURRENT_H
#include "../types.h"
#include "../size_bgru.h"

// Taken from: q_bidirectional_gru_forward_recurrent_kernel.h
const gru_weigth_t g_0_forward_kernel_recurrent[GRU_RKERNEL_SIZE] = {
    0.9876484274864197,
    -0.06075083464384079,
    0.26505231857299805,
};

#endif // GRU_0_FORWARD_KERNEL_RECURRENT_H