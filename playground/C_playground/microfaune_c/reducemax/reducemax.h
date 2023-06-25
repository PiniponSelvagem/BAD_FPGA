#pragma once

#ifndef REDUCEMAX_H
#define REDUCEMAX_H

#include "../types.h"
#include <limits.h>

template <int RM_IN_LINES, int RM_IN_COLS, int RM_OUT_LINES, int RM_OUT_COLS>
void reducemax_0(
    const reducemax_t input[CHANNELS][RM_IN_LINES][RM_IN_COLS],
    reducemax_t output[RM_OUT_LINES][RM_OUT_COLS]
) {
    for (int c = 0; c < CHANNELS; ++c) {
        for (int row = 0; row < RM_IN_LINES; ++row) {
            reducemax_t maxval = INT_MIN;
            for (int col = 0; col < RM_IN_COLS; ++col) {
                if (input[c][row][col] > maxval) {
                    maxval = input[c][row][col];
                }
            }
            output[c][row] = maxval;
        }
    }
}

template <int RM_IN_LINES>
void reducemax_1(const reducemax_t input[RM_IN_LINES], reducemax_t output[1]) {
    reducemax_t maxval = INT_MIN;
    for (int row = 0; row < RM_IN_LINES; ++row) {
        if (input[row] > maxval) {
            maxval = input[row];
        }
    }
    output[0] = maxval;
}


#endif // REDUCEMAX_H