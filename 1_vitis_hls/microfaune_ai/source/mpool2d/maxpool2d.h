#pragma once

#ifndef MP2D_H
#define MP2D_H

#include "../global_settings.h"
#include "../types.h"

typedef ap_uint<9> mpool_row_t;
typedef ap_uint<6> mpool_col_t;
typedef ap_uint<1> mpool_save_t;

void maxpool2d(
    const mpool_col_t inCols,
    const mpool_save_t saveOffset,
    mpool_t* input_output
) {
    #define MP2D_STRIDE 2
    MPOOL_loop_channel: for (i64_t c = 0; c < CHANNELS; ++c) {
        mpool_t* pinput = input_output + (c * CNN_LINES_PAD * CNN_COLS_PAD);
        mpool_t* poutput = pinput;
        MPOOL_loop_row: for (mpool_row_t row = PADDING_OFFSET, orow = saveOffset; row < (CNN_LINES_PAD - PADDING_OFFSET); ++row, ++orow) {
            mpool_t* pinput_row  = pinput + (row * CNN_COLS_PAD);
            mpool_t* poutput_row = poutput + (orow * CNN_COLS_PAD);
            mpool_col_t ocol = saveOffset;
            MPOOL_loop_col: for (mpool_col_t col = PADDING_OFFSET; col < (inCols - PADDING_OFFSET); col += MP2D_STRIDE, ++ocol) {
                mpool_t* pinout_col = pinput_row + col;
                mpool_t* poutput_col = poutput_row + ocol;
                
                mpool_t maxval;
                mpool_t in = *pinout_col; //input[row][col];
                mpool_t in_next = *(pinout_col+1); //input[row][col + 1];

                if (in > in_next) {
                    maxval = in;
                }
                else {
                    maxval = in_next;
                }

                *poutput_col = maxval;
            }
            //clear last column to fix padding
            mpool_t* poutput_col_end = poutput_row + ocol;
            *poutput_col_end = 0;
        }
    }
}


#endif // MP2D_H