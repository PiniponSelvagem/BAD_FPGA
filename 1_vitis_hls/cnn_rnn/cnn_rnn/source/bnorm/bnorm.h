#pragma once

#ifndef BNORM_H
#define BNORM_H

#include "bnorm_settings.h"
#include <math.h>

typedef ap_uint<9> bnorm_row_t;
typedef ap_uint<6> bnrom_col_t;

template <int BN_LINES, int BN_COLS>
void bnorm(
    const bnorm_t input[BN_LINES][BN_COLS],
    const bnorm_t gamma,
    const bnorm_t beta,
    const bnorm_t movingmean,
    const bnorm_t movingvariance,
    bnorm_t output[BN_LINES][BN_COLS]
) {
    const bnorm_t epsilon = BNORM_EPSILON;
    BNORM_loop_row: for (bnorm_row_t row = PADDING_OFFSET; row < (BN_LINES - PADDING_OFFSET); ++row) {
        BNORM_loop_col: for (bnrom_col_t col = PADDING_OFFSET; col < (BN_COLS - PADDING_OFFSET); ++col) {
            bnorm_t sqrt_value = movingvariance + epsilon;
            bnorm_t normalized = (input[row][col] - movingmean) / (bnorm_t)(sqrt((float)sqrt_value));
            bnorm_t out = gamma * normalized + beta;

            if (out < 0) { out = 0; } // ReLu
            output[row][col] = out;
        }
    }
}


#endif // BNORM_H