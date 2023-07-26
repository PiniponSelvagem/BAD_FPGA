#pragma once

#ifndef MP2D_H
#define MP2D_H

#include "../global_settings.h"
#include "../types.h"

#ifdef __VITIS_HLS__
typedef ap_uint<9> mpool_row_t;
typedef ap_uint<6> mpool_col_t;
typedef ap_uint<1> mpool_save_t;

typedef ap_uint<21> mpool_p_inout;
#endif
#ifdef _MSC_VER
typedef int mpool_row_t;
typedef int mpool_col_t;
typedef int mpool_save_t;

typedef int mpool_p_inout;
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
        mpool_p_inout pinput_offset_channel  = (c * CNN_LINES_PAD * inCols);
        mpool_p_inout poutput_offset_channel = (c * cnnLines * outCols);
        MPOOL_loop_row: for (mpool_row_t row = PADDING_OFFSET, orow = saveOffset; row < (CNN_LINES_PAD - PADDING_OFFSET); ++row, ++orow) {
            mpool_p_inout pinput_offset_row  = pinput_offset_channel + (row * inCols);
            mpool_p_inout poutput_offset_row = poutput_offset_channel + (orow * outCols);
            mpool_col_t ocol = saveOffset;
            MPOOL_loop_col: for (mpool_col_t col = PADDING_OFFSET; col < (inCols - PADDING_OFFSET); col += MP2D_STRIDE, ++ocol) {
                mpool_t* pinout_col  = input_output + pinput_offset_row + col;
                mpool_t* poutput_col = input_output + poutput_offset_row + ocol;
                
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
                mpool_t* poutput_row = input_output + poutput_offset_row;
                *poutput_row = 0;
                mpool_t* poutput_col_end = poutput_row + ocol;
                *poutput_col_end = 0;
            }
        }

        // clear first and last row to fix padding
        if (saveOffset != 0) {
            mpool_t* poutput_clear_start = input_output + poutput_offset_channel;
            mpool_t* poutput_clear_end = input_output + poutput_offset_channel + ((cnnLines - 1) * outCols);
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

#endif // MP2D_H