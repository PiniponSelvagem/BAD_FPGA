#pragma once

#ifndef MP2D_H
#define MP2D_H

#include "../global_settings.h"
#include <limits.h>

template <int MP_IN_LINES, int MP_IN_COLS, int MP_OUT_LINES, int MP_OUT_COLS, int SAVE_OFFSET>
void maxpool2d(
    const mpool_t input[MP_IN_LINES][MP_IN_COLS],
    mpool_t output[MP_OUT_LINES][MP_OUT_COLS]
) {
    #define MP2D_STRIDE 2
    for (int row = PADDING_OFFSET, orow = SAVE_OFFSET; row < (MP_IN_LINES - PADDING_OFFSET); ++row, ++orow) {
        for (int col = PADDING_OFFSET, ocol = SAVE_OFFSET; col < (MP_IN_COLS - PADDING_OFFSET); col += MP2D_STRIDE, ++ocol) {
            mpool_t maxval;
            mpool_t in = input[row][col];
            mpool_t in_next = input[row][col + 1];

            if (in > in_next) {
                maxval = in;
            }
            else {
                maxval = in_next;
            }

            output[orow][ocol] = maxval;
        }
    }
}


#endif // MP2D_H