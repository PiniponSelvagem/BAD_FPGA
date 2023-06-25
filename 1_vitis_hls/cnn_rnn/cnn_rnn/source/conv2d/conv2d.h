#pragma once

#ifndef CONV2D_H
#define CONV2D_H

#include <ap_int.h>
#include "conv2d_settings.h"
#include "data/conv2d_0.h"

typedef ap_uint<9> conv_row_t;
typedef ap_uint<6> conv_col_t;
typedef ap_uint<2> conv_k_t;

template <int C_IN_LINES, int C_IN_COLS, int C_OUT_LINES, int C_OUT_COLS>
void conv2d(
    const conv_t input[C_IN_LINES][C_IN_COLS],
    const conv_t kernel[C2D_KERNEL_LINES][C2D_KERNEL_COLS],
    const conv_t bias,
    conv_t output[C_OUT_LINES][C_OUT_COLS]
) {
    CONV_loop_row: for (conv_row_t orow = PADDING_OFFSET; orow < (C_OUT_LINES - PADDING_OFFSET); ++orow) {
    	CONV_loop_col: for (conv_col_t ocol = PADDING_OFFSET; ocol < (C_OUT_COLS - PADDING_OFFSET); ++ocol) {
#pragma HLS PIPELINE
            conv_t acc = bias;
            conv_t acc_sat;
            conv_row_t korow = orow - PADDING_OFFSET;
            CONV_loop_k1: for (conv_k_t krow = 0; krow < C2D_KERNEL_LINES; ++krow, ++korow) {
//#pragma HLS PIPELINE
                conv_col_t kocol = ocol - PADDING_OFFSET;
            	CONV_loop_k2: for (conv_k_t kcol = 0; kcol < C2D_KERNEL_COLS; ++kcol, ++kocol) {
//#pragma HLS PIPELINE
                    acc += kernel[krow][kcol] * input[korow][kocol];
                }
            }

            /*
            if (acc > 255)
                acc_sat = 255;
            else if (acc < 0)
                acc_sat = 0;    // ReLu
            else
            */
            acc_sat = acc;
            output[orow][ocol] = acc_sat;
        }
    }
}


template <int C_IN_LINES, int C_IN_COLS, int C_OUT_LINES, int C_OUT_COLS>
void conv2d_multi(
    const conv_t input[C_IN_LINES][C_IN_COLS],
    const conv_t prev[C_IN_LINES][C_IN_COLS],
    const conv_t kernel[C2D_KERNEL_LINES][C2D_KERNEL_COLS],
    conv_t output[C_OUT_LINES][C_OUT_COLS]
) {
    CONV_M_loop_row: for (conv_row_t orow = PADDING_OFFSET; orow < (C_OUT_LINES - PADDING_OFFSET); ++orow) {
        CONV_M_loop_col: for (conv_col_t ocol = PADDING_OFFSET; ocol < (C_OUT_COLS - PADDING_OFFSET); ++ocol) {
#pragma HLS PIPELINE
            conv_t acc = 0;
            conv_t acc_sat;
            conv_row_t korow = orow - PADDING_OFFSET;
            CONV_loop_k1: for (conv_k_t krow = 0; krow < C2D_KERNEL_LINES; ++krow, ++korow) {
//#pragma HLS PIPELINE
                conv_col_t kocol = ocol - PADDING_OFFSET;
            	CONV_loop_k2: for (conv_k_t kcol = 0; kcol < C2D_KERNEL_COLS; ++kcol, ++kocol) {
//#pragma HLS PIPELINE
                    acc += kernel[krow][kcol] * input[korow][kocol];
                }
            }

            /*
            if (acc > 255)
                acc_sat = 255;
            else if (acc < 0)
                acc_sat = 0;    // ReLu
            else
            */
            acc_sat = acc + prev[orow][ocol];
            output[orow][ocol] = acc_sat;
        }
    }
}

/*
void conv2d_postprocess(const conv_t prepro[C2D_0__IN_LINES][C2D_0__IN_COLS], conv_t output[INPUT_LINES][INPUT_COLS]) {
    for (int h = 0; h < (C2D_0__IN_LINES - 1); ++h) {
        for (int w = 0; w < (C2D_0__IN_COLS - 1); ++w) {
            if (h == 0 || h == (C2D_0__IN_LINES - 1) || w == 0 || w == (C2D_0__IN_COLS - 1)) {
                ;
            }
            else {
                output[h - PADDING_OFFSET][w - PADDING_OFFSET] = prepro[h][w];
            }
        }
    }
}
*/


#endif // CONV2D_H
