#pragma once

#ifndef MP2D_H
#define MP2D_H

#include <limits.h>

template <int MP_IN_LINES, int MP_IN_COLS, int MP_OUT_LINES, int MP_OUT_COLS>
void maxpool2d(
    const mpool_t input[MP_IN_LINES][MP_IN_COLS],
    mpool_t output[MP_OUT_LINES][MP_OUT_COLS]
) {
    #define MP2D_STRIDE 2
    for (int row = PADDING_OFFSET; row < (MP_IN_LINES - PADDING_OFFSET); ++row) {
        int ocol = PADDING_OFFSET;
        for (int col = PADDING_OFFSET; col < (MP_IN_COLS - PADDING_OFFSET); col += MP2D_STRIDE) {
            mpool_t maxval;

            if (input[row][col] > input[row][col+1]) {
                maxval = input[row][col];
            }
            else {
                maxval = input[row][col+1];
            }

            output[row][ocol++] = maxval;
        }
    }
}


#endif // MP2D_H