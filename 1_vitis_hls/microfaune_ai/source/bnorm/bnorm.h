#pragma once

#ifndef BNORM_H
#define BNORM_H

#include "bnorm_settings.h"
#include <math.h>

typedef ap_uint<9> bnorm_row_t;
typedef ap_uint<6> bnrom_col_t;

typedef ap_uint<22> bnorm_c_l_c;
typedef ap_uint<22> bnorm_offset_col;

void bnorm(
    const i64_t channels,
    const bnorm_t inLines,
    const bnorm_t inCols,
    bnorm_t* input_output,
    const bnorm_t gamma[CHANNELS],
    const bnorm_t beta[CHANNELS],
    const bnorm_t movingmean[CHANNELS],
    const bnorm_t movingvariance[CHANNELS]
) {
    const bnorm_t epsilon = BNORM_EPSILON;
    BNORM_loop_channel: for (i64_t c = 0; c < channels; ++c) {
        bnorm_c_l_c output_offset_c = c * inLines * inCols;
        BNORM_loop_row: for (bnorm_row_t row = PADDING_OFFSET; row < (inLines - PADDING_OFFSET); ++row) {
            BNORM_loop_col: for (bnrom_col_t col = PADDING_OFFSET; col < (inCols - PADDING_OFFSET); ++col) {
                bnorm_offset_col offsetcol = (output_offset_c + row * inCols + col); // no idea why i have to do this wierd dance and cant combine in pinout, but compiler didnt liked it
                bnorm_t* pinout = (input_output + offsetcol);
                bnorm_t value = movingvariance[c] + epsilon;
                #ifdef USE_FLOAT
                bnorm_t normalized = (*pinout - movingmean[c]) / (bnorm_t)(sqrt(value));
                #else
                bnorm_t normalized = (*pinout - movingmean[c]) / (bnorm_t)(sqrt(value.to_float()));
                #endif
                bnorm_t out = gamma[c] * normalized + beta[c];

                if (out < 0) { out = 0; } // ReLu
                *pinout = out;
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
    const bnorm_t epsilon = BNORM_EPSILON;
    BNORM_loop_row: for (bnorm_row_t row = PADDING_OFFSET; row < (BN_LINES - PADDING_OFFSET); ++row) {
        BNORM_loop_col: for (bnrom_col_t col = PADDING_OFFSET; col < (BN_COLS - PADDING_OFFSET); ++col) {
            bnorm_t sqrt_value = movingvariance + epsilon;
            #ifdef USE_FLOAT
            bnorm_t normalized = (input[row][col] - movingmean) / (bnorm_t)(sqrt(sqrt_value));
            #else
            bnorm_t normalized = (input[row][col] - movingmean) / (bnorm_t)(sqrt(sqrt_value.to_float()));
            #endif
            bnorm_t out = gamma * normalized + beta;

            if (out < 0) { out = 0; } // ReLu
            output[row][col] = out;
        }
    }
}


#endif // BNORM_H