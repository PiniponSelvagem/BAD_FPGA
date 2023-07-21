#pragma once

#ifndef TIMEDIST_H
#define TIMEDIST_H

#include "timedist_settings.h"


template <
    int TD_IN_LINES, int TD_IN_COLS,
    int TD_KERNEL_LINES, int TD_KERNEL_COLS,
    int TD_BIAS_SIZE,
    int TD_OUT_LINES, int TD_OUT_COLS
>
void timedistributed_dense(
    const timedist_t input[TD_IN_COLS],
    const timedist_t kernel[TD_KERNEL_LINES][TD_KERNEL_COLS],
    const timedist_t bias[TD_BIAS_SIZE],
    timedist_t output[TD_OUT_COLS]
) {
    for (int ocol = 0; ocol < TD_OUT_COLS; ++ocol) {
        timedist_t acc = bias[ocol];
        for (int kcol = 0; kcol < TD_KERNEL_COLS; ++kcol) {
            timedist_t k = kernel[ocol][kcol];
            timedist_t i = input[kcol];
            acc += k * i;
        }
        output[ocol] = SIGMOID(acc);
    }
}


#endif // TIMEDIST_H