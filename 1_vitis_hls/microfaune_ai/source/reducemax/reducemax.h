#pragma once

#ifndef REDUCEMAX_H
#define REDUCEMAX_H

#include "../global_settings.h"
#include "../types.h"

#define RMAX_MIN_VALUE  0

typedef ap_uint<7> rmax_cha_t;
typedef ap_uint<9> rmax_row_t;
typedef ap_uint<6> rmax_col_t;

void reducemax_0_saveTranspose(
    reducemax_t* input_output
) {
    RMAX_0_loop_channel: for (rmax_cha_t c = 0; c < CHANNELS; ++c) {
        RMAX_0_loop_row: for (rmax_row_t row = 0; row < CNN_LINES; ++row) {
            reducemax_t maxval = RMAX_MIN_VALUE;
            RMAX_0_loop_col: for (rmax_col_t col = 0; col < RMAX_0__IN_COLS; ++col) {
                reducemax_t* input_offset = input_output + (c * CNN_LINES_PAD * CNN_COLS_PAD) + (row * CNN_COLS_PAD) + col;
                if (*input_offset > maxval) {
                    maxval = *input_offset;
                }
            }
            reducemax_t* poutput = input_output + (row * CHANNELS) + c;
            *poutput = maxval;
        }
    }
}


template <int RM_IN_LINES>
void reducemax_1(const output_t input[RM_IN_LINES], output_t output[1]) {
    output_t maxval = RMAX_MIN_VALUE;
    RMAX_1_loop_row: for (rmax_row_t row = 0; row < RM_IN_LINES; ++row) {
        if (input[row] > maxval) {
            maxval = input[row];
        }
    }
    output[0] = maxval;
}


#endif // REDUCEMAX_H
