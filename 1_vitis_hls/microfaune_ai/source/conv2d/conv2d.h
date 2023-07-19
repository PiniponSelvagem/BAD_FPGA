#pragma once

#ifndef CONV2D_H
#define CONV2D_H

#include "conv2d_settings.h"
#include "data/conv2d_0.h"

#ifdef __VITIS_HLS__
#include <ap_int.h>
typedef ap_uint<9> conv_row_t;
typedef ap_uint<6> conv_col_t;
typedef ap_uint<2> conv_k_t;

typedef ap_uint<22> conv_c_l_c;
#endif
#ifdef _MSC_VER
typedef int conv_row_t;
typedef int conv_col_t;
typedef int conv_k_t;

typedef int conv_c_l_c;
#endif


// aux array in filter calculations, must have the size of the max input ever provided
conv_t aux[CHANNELS][CNN_LINES_PAD][CNN_COLS_PAD];
conv_t* prev = (conv_t*)aux;

void conv2d(
    const i64_t filters,
    const conv_col_t inoutCols,
    const conv_t* input,        // 3D array
    const conv_t* kernel,       // 3D array if filters==1, 4D array if filters>1
    const conv_t* bias,         // 1D array
    conv_t* output
) {
    // clear all prev and output
    // TODO: improve so it only clears necessary parts of the arrays
    for (i64_t c = 0; c < CHANNELS; ++c) {
        for (conv_row_t row = 0; row < CNN_LINES_PAD; ++row) {
            for (conv_col_t col = 0; col < CNN_COLS_PAD; ++col) {
                aux[c][row][col] = 0;
                conv_t* poutput = output + (c * CNN_LINES_PAD * inoutCols) + (row * inoutCols) + col;
                *poutput = 0;
            }
        }
    }

    conv_t bias_none = 0;
    conv_t* pbias_backup = (conv_t*)bias;
    conv_t* pbias = pbias_backup;

    CONV_loop_filter: for (i64_t f = 0; f < filters; ++f) {
        conv_t* pkernel = (conv_t*)kernel + (f * CHANNELS * C2D_KERNEL_LINES * C2D_KERNEL_COLS);

        conv_c_l_c output_offset;
        conv_t* pprev_offset;
        conv_t* poutput_offset;
        if (filters > 1) {
            pprev_offset = prev + f * CNN_LINES_PAD * CNN_COLS_PAD;
            poutput_offset = output + f * CNN_LINES_PAD * inoutCols;
        }

        CONV_loop_channel: for (i64_t c = 0; c < CHANNELS; ++c) {
            conv_t* pkernel_c = pkernel + (c * C2D_KERNEL_LINES * C2D_KERNEL_COLS);
            conv_t* pinput_c = (conv_t*)input + (c * CNN_LINES_PAD * inoutCols);

            if (filters == 1) {
                pprev_offset = prev + c * CNN_LINES_PAD * CNN_COLS_PAD;
                poutput_offset = output + c * CNN_LINES_PAD * inoutCols;
            }

            CONV_loop_row: for (conv_row_t orow = PADDING_OFFSET; orow < (CNN_LINES_PAD - PADDING_OFFSET); ++orow) {
                conv_t* pprev_offset_orow = pprev_offset + (orow * CNN_COLS_PAD);
                conv_t* poutput_offset_orow = poutput_offset + (orow * inoutCols);
                CONV_loop_col: for (conv_col_t ocol = PADDING_OFFSET; ocol < (inoutCols - PADDING_OFFSET); ++ocol) {
        #pragma HLS PIPELINE
                    conv_t acc = *pbias;
                    conv_t acc_sat;
                    conv_row_t korow = orow - PADDING_OFFSET;
                    CONV_loop_k1: for (conv_k_t krow = 0; krow < C2D_KERNEL_LINES; ++krow, ++korow) {
        //#pragma HLS PIPELINE
                        conv_t* pkernel_c_krow = pkernel_c + (krow * C2D_KERNEL_COLS);
                        conv_t* pinput_c_krow = pinput_c + (korow * inoutCols);
                        conv_col_t kocol = ocol - PADDING_OFFSET;
                        CONV_loop_k2: for (conv_k_t kcol = 0; kcol < C2D_KERNEL_COLS; ++kcol, ++kocol) {
        //#pragma HLS PIPELINE
                            //conv_t k = kernelArr[f][c][krow][kcol];
                            conv_t k = *(pkernel_c_krow + kcol);
                            //conv_t i = input[c][korow][kocol];
                            conv_t i = *(pinput_c_krow + kocol);
                            acc += k * i;
                        }
                    }

// TODO: Uncomment after removing BatchNormalization layer
// TODO: Might require adjustments since max value might no longer be 255
                    /*
                    if (acc > 255)
                        acc_sat = 255;
                    else if (acc < 0)
                        acc_sat = 0;    // ReLu
                    else
                    */
                    
                    //output[orow][ocol] = acc_sat;
                    conv_t* poutput = (poutput_offset_orow + ocol);
                    conv_t* pprev = (pprev_offset_orow + ocol);
                    acc_sat = acc + *pprev;
                    *poutput = acc_sat;
                    if ((c == 0 && filters == 1) || (filters > 1))
                        *pprev = acc_sat;
                }
            }
            if (filters == 1) {
                // Loop "channels" only advances bias if only 1 "filters"
                *pbias++;
            }
            else if (c >= 0) {
                // If "filters > 1", only 1st channel should have the bias value, others 0
                pbias = &bias_none;
            }
        }
        // Set the pbias from bias_NONE to bias
        pbias = (conv_t*)++pbias_backup;
    }
}





template <int C_IN_LINES, int C_IN_COLS, int C_OUT_LINES, int C_OUT_COLS>
void conv2d_old(
    const conv_t input[C_IN_LINES][C_IN_COLS],
    const conv_t kernel[C2D_KERNEL_LINES][C2D_KERNEL_COLS],
    const conv_t bias,
    conv_t output[C_OUT_LINES][C_OUT_COLS]
) {
    for (int orow = PADDING_OFFSET; orow < (C_OUT_LINES - PADDING_OFFSET); ++orow) {
        for (int ocol = PADDING_OFFSET; ocol < (C_OUT_COLS - PADDING_OFFSET); ++ocol) {
            conv_t acc = bias;
            conv_t acc_sat;
            for (int krow = 0, korow = orow - PADDING_OFFSET; krow < C2D_KERNEL_LINES; ++krow, korow++) {
                for (int kcol = 0, kocol = ocol - PADDING_OFFSET; kcol < C2D_KERNEL_COLS; ++kcol, kocol++) {
                    conv_t k = kernel[krow][kcol];
                    conv_t i = input[korow][kocol];
                    acc += k * i;
                }
            }
            /*
            for (int krow = 0; krow < C2D_KERNEL_LINES; ++krow) {
                for (int kcol = 0; kcol < C2D_KERNEL_COLS; ++kcol) {
                    acc += kernel[krow][kcol] * input[orow + krow - PADDING_OFFSET][ocol + kcol - PADDING_OFFSET];
                }
            }
            */

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

#include <stdio.h>
template <int C_IN_LINES, int C_IN_COLS, int C_OUT_LINES, int C_OUT_COLS>
void conv2d_multi(
    const conv_t input[C_IN_LINES][C_IN_COLS],
    const conv_t prev[C_IN_LINES][C_IN_COLS],
    const conv_t kernel[C2D_KERNEL_LINES][C2D_KERNEL_COLS],
    conv_t output[C_OUT_LINES][C_OUT_COLS]
) {
    for (int orow = PADDING_OFFSET; orow < (C_OUT_LINES - PADDING_OFFSET); ++orow) {
        for (int ocol = PADDING_OFFSET; ocol < (C_OUT_COLS - PADDING_OFFSET); ++ocol) {
            conv_t acc = 0;
            conv_t acc_sat;
            /*
            conv_t acc_sat;
            for (int krow = 0; krow < C2D_KERNEL_LINES; ++krow) {
                for (int kcol = 0; kcol < C2D_KERNEL_COLS; ++kcol) {
                    conv_t wg = kernel[krow][kcol];
                    conv_t in = input[orow + krow - PADDING_OFFSET][ocol + kcol - PADDING_OFFSET];
                    acc += wg * in;
                }
            }
            */
            for (int krow = 0, korow = orow - PADDING_OFFSET; krow < C2D_KERNEL_LINES; ++krow, korow++) {
                for (int kcol = 0, kocol = ocol - PADDING_OFFSET; kcol < C2D_KERNEL_COLS; ++kcol, kocol++) {
                    conv_t k = kernel[krow][kcol];
                    conv_t i = input[korow][kocol];
                    acc += k * i;
                }
            }

            /*
            if (acc > 255)
                acc_sat = 255;
            else if (acc < 0)
                acc_sat = 0;    // ReLu
            else
            */
            conv_t pv = prev[orow][ocol];
            acc_sat = acc + pv;
            output[orow][ocol] = acc_sat;
        }
    }
}


#endif // CONV2D_H
