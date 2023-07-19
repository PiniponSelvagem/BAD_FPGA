#pragma once

#ifndef MP2D_H
#define MP2D_H

#include "../global_settings.h"
#include "../types.h"

#ifdef __VITIS_HLS__
typedef ap_uint<9> mpool_row_t;
typedef ap_uint<6> mpool_col_t;
typedef ap_uint<1> mpool_save_t;
#endif
#ifdef _MSC_VER
typedef int mpool_row_t;
typedef int mpool_col_t;
typedef int mpool_save_t;
#endif

void maxpool2d(
    const mpool_col_t inCols,
    const mpool_col_t outCols,
    const mpool_save_t saveOffset,
    mpool_t* input_output
) {
    i512_t cnnLines;
    if (saveOffset != 0)
        cnnLines = CNN_LINES_PAD;
    else
        cnnLines = CNN_LINES;

    #define MP2D_STRIDE 2
    MPOOL_loop_channel: for (i64_t c = 0; c < CHANNELS; ++c) {
        mpool_t* pinput  = input_output + (c * CNN_LINES_PAD * inCols);
        mpool_t* poutput = input_output + (c * cnnLines * outCols);
        MPOOL_loop_row: for (mpool_row_t row = PADDING_OFFSET, orow = saveOffset; row < (CNN_LINES_PAD - PADDING_OFFSET); ++row, ++orow) {
            mpool_t* pinput_row  = pinput + (row * inCols);
            mpool_t* poutput_row = poutput + (orow * outCols);
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
            // clear first and last column to fix padding
            if (saveOffset != 0) {
                *poutput_row = 0;
                mpool_t* poutput_col_end = poutput_row + ocol;
                *poutput_col_end = 0;
            }
        }

        // clear first and last row to fix padding
        if (saveOffset != 0) {
            mpool_t* poutput_clear_start = poutput;
            mpool_t* poutput_clear_end = poutput + ((cnnLines - 1) * outCols);
            for (int ocol = 0; ocol < outCols; ++ocol) {
                // add padding top
                *poutput_clear_start = 0;
                ++poutput_clear_start;
                // add padding bottom
                *poutput_clear_end = 0;
                ++poutput_clear_end;
            }
        }
    }
}



template <int MP_IN_LINES, int MP_IN_COLS, int MP_OUT_LINES, int MP_OUT_COLS, int SAVE_OFFSET>
void maxpool2d_old(
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