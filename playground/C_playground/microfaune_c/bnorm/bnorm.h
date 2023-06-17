#pragma once
#include "bnorm_settings.h"

#include <math.h>

#ifndef BNORM_H
#define BNORM_H

template <int BN_LINES, int BN_COLS>
void bnorm(
    const bnorm_t input[BN_LINES][BN_COLS],
    const bnorm_t gamma,
    const bnorm_t beta,
    const bnorm_t movingmean,
    const bnorm_t movingvariance,
    bnorm_t output[BN_LINES][BN_COLS]
) {
    for (int row = PADDING_OFFSET; row < (BN_LINES - PADDING_OFFSET); ++row) {
        for (int col = PADDING_OFFSET; col < (BN_COLS - PADDING_OFFSET); ++col) {
            bnorm_t normalized = (input[row][col] - movingmean) / sqrt(movingvariance + BNORM_EPSILON);
            bnorm_t out = gamma * normalized + beta;

            if (out < 0) { out = 0; } // ReLu
            output[row][col] = out;
        }
    }
}


#endif // BNORM_H