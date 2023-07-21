#pragma once

#ifndef REDUCEMAX_H
#define REDUCEMAX_H

#include "../global_settings.h"
#include "../types.h"

#define RMAX_MIN_VALUE  0

#ifdef __VITIS_HLS__
typedef ap_uint<7> rmax_cha_t;
typedef ap_uint<9> rmax_row_t;
typedef ap_uint<6> rmax_col_t;
#endif
#ifdef _MSC_VER
typedef int rmax_cha_t;
typedef int rmax_row_t;
typedef int rmax_col_t;
#endif

void reducemax_0_saveTranspose(
    reducemax_t* input,
    reducemax_t* output
) {
    RMAX_0_loop_channel: for (rmax_cha_t c = 0; c < CHANNELS; ++c) {
        reducemax_t* input_offset = input + (c * CNN_LINES * RMAX_0__IN_COLS);
        RMAX_0_loop_row: for (rmax_row_t row = 0; row < CNN_LINES; ++row) {
            reducemax_t maxval = RMAX_MIN_VALUE;
            reducemax_t* input_offset_row = input_offset + (row * RMAX_0__IN_COLS);
            reducemax_t* output_offset = output + (row * CHANNELS);
            RMAX_0_loop_col: for (rmax_col_t col = 0; col < RMAX_0__IN_COLS; ++col) {
                reducemax_t value = *(input_offset_row + col);
                if (value > maxval) {
                    maxval = value;
                }
            }
            reducemax_t* poutput = output_offset + c;
            *poutput = maxval;
        }
    }
}

void reducemax_1(
    const output_t* input,
    output_t* output
) {
    output_t maxval = RMAX_MIN_VALUE;
    RMAX_1_loop_row: for (rmax_row_t row = 0; row < RNN_LINES_GRU; ++row) {
        output_t value = *(input + row);
        if (value > maxval) {
            maxval = value;
        }
    }
    *output = maxval;
}

#endif // REDUCEMAX_H
