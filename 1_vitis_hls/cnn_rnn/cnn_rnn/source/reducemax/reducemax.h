#pragma once

#ifndef REDUCEMAX_H
#define REDUCEMAX_H

#include "../global_settings.h"
#include "../types.h"

#define RMAX_MIN_VALUE  0

typedef ap_uint<7> rmax_cha_t;
typedef ap_uint<9> rmax_row_t;
typedef ap_uint<6> rmax_col_t;

template <int RM_IN_LINES, int RM_IN_COLS, int RM_OUT_LINES, int RM_OUT_COLS>
void reducemax_0_saveTranspose(
    const reducemax_t input[CHANNELS][RM_IN_LINES][RM_IN_COLS],
    reducemax_t output[RM_OUT_LINES][RM_OUT_COLS]
) {
    RMAX_0_loop_channel: for (rmax_cha_t c = 0; c < CHANNELS; ++c) {
        RMAX_0_loop_row: for (rmax_row_t row = 0; row < RM_IN_LINES; ++row) {
            reducemax_t maxval = RMAX_MIN_VALUE;
            RMAX_0_loop_col: for (rmax_col_t col = 0; col < RM_IN_COLS; ++col) {
                if (input[c][row][col] > maxval) {
                    maxval = input[c][row][col];
                }
            }
            output[row][c] = maxval;
        }
    }
}

template <int RM_IN_LINES>
void reducemax_1(const reducemax_t input[RM_IN_LINES], reducemax_t output[1]) {
    reducemax_t maxval = RMAX_MIN_VALUE;
    RMAX_1_loop_row: for (rmax_row_t row = 0; row < RM_IN_LINES; ++row) {
        if (input[row] > maxval) {
            maxval = input[row];
        }
    }
    output[0] = maxval;
}


#endif // REDUCEMAX_H
