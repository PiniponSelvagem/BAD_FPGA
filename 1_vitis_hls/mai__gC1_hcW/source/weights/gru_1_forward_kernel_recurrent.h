#ifndef GRU_1_FORWARD_KERNEL_RECURRENT_H
#define GRU_1_FORWARD_KERNEL_RECURRENT_H
#include "../types.h"
#include "../size_bgru.h"

// Taken from: q_bidirectional_1_gru_forward_recurrent_kernel.h
const gru_weigth_t g_1_forward_kernel_recurrent[GRU_RKERNEL_SIZE] = {
    0.6167978048324585,
    1.1926499605178833,     // This will be capped
    1.0230984687805176,     // This will be capped
};

#endif // GRU_1_FORWARD_KERNEL_RECURRENT_H