#pragma once

#ifndef BNORM_H
#define BNORM_H

#include "bnorm_settings.h"
#include <math.h>

#ifdef __VITIS_HLS__
typedef ap_uint<9> bnorm_row_t;
typedef ap_uint<6> bnrom_col_t;

typedef ap_uint<22> bnorm_c_l_c;
typedef ap_uint<22> bnorm_offset_col;
#endif
#ifdef _MSC_VER
typedef int bnorm_row_t;
typedef int bnrom_col_t;

typedef int bnorm_c_l_c;
typedef int bnorm_offset_col;
#endif


void bnorm(
    const bnrom_col_t inCols,
    bnorm_t* input_output,
    const bnorm_t gamma[CHANNELS],
    const bnorm_t beta[CHANNELS],
    const bnorm_t movingmean[CHANNELS],
    const bnorm_t movingvariance[CHANNELS]
) {
    const bnorm_t epsilon = BNORM_EPSILON;
    BNORM_loop_channel: for (i64_t c = 0; c < CHANNELS; ++c) {
        bnorm_t* pinout = input_output + (c * CNN_LINES_PAD * inCols);
        BNORM_loop_row: for (bnorm_row_t row = PADDING_OFFSET; row < (CNN_LINES_PAD - PADDING_OFFSET); ++row) {
            bnorm_t* pinout_row = pinout + (row * inCols);
            BNORM_loop_col: for (bnrom_col_t col = PADDING_OFFSET; col < (inCols - PADDING_OFFSET); ++col) {
                bnorm_t* pinout_col = pinout_row + col;
                bnorm_t inValue = *pinout_col;
                bnorm_t value = movingvariance[c] + BNORM_EPSILON;
                bnorm_t normalized = (inValue - movingmean[c]) / sqrt(movingvariance[c] + BNORM_EPSILON);
                bnorm_t out = gamma[c] * normalized + beta[c];

                if (out < 0) { out = 0; } // ReLu
                *pinout_col = out;
            }
        }
    }
}


template <int BN_LINES, int BN_COLS>
void bnorm_old(
    const bnorm_t input[BN_LINES][BN_COLS],
    const bnorm_t gamma,
    const bnorm_t beta,
    const bnorm_t movingmean,
    const bnorm_t movingvariance,
    bnorm_t output[BN_LINES][BN_COLS]
) {
    for (int row = PADDING_OFFSET; row < (BN_LINES - PADDING_OFFSET); ++row) {
        for (int col = PADDING_OFFSET; col < (BN_COLS - PADDING_OFFSET); ++col) {
            bnorm_t inValue = input[row][col];
            bnorm_t normalized = (inValue - movingmean) / sqrt(movingvariance + BNORM_EPSILON);
            bnorm_t out = gamma * normalized + beta;

            if (out < 0) { out = 0; } // ReLu
            output[row][col] = out;
        }
    }
}


#endif // BNORM_H