#ifndef GRU_1_BACKWARD_KERNEL_H
#define GRU_1_BACKWARD_KERNEL_H
#include "../types.h"
#include "../size_bgru.h"

// Taken from: q_bidirectional_1_gru_backward_kernel.h
const gru_weigth_t g_1_backward_kernel[GRU1_KERNEL_SIZE] = {
    0.09664391726255417,
    0.5056001543998718,
    0.8859548568725586,
    -0.8968144059181213,
    -0.37502527236938477,
    -1.2637395858764648,
};

#endif // GRU_1_BACKWARD_KERNEL_H