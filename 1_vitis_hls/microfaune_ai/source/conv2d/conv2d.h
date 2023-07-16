#pragma once

#ifndef CONV2D_H
#define CONV2D_H

#include <ap_int.h>
#include "conv2d_settings.h"
#include "data/conv2d_0.h"

typedef ap_uint<9> conv_row_t;
typedef ap_uint<6> conv_col_t;
typedef ap_uint<2> conv_k_t;

typedef ap_uint<22> conv_c_l_c;


// aux array in filter calculations, must have the size of the max input ever provided
conv_t aux[CHANNELS][CNN_LINES_PAD][CNN_COLS_PAD];
conv_t* prev = (conv_t*)aux;

void conv2d(
    const i64_t filters,
    const conv_col_t inCols,
    const conv_col_t outCols,
    const conv_t* input,        // 3D array
    const conv_t* kernel,       // 3D array if filters==1, 4D array if filters>1
    const conv_t* bias,         // 1D array
    conv_t* output
) {

    // aux array: set usable bottom line to 0 because maxpool will reduce lines and old aux data is still there
    for (i64_t c = 0; c < CHANNELS; ++c) {
        conv_t* pprev = prev + (c * CNN_LINES_PAD * CNN_COLS_PAD) + (inCols-1);
        conv_t* poutput = output + (c * CNN_LINES_PAD * CNN_COLS_PAD) + (inCols-1);
        for (conv_row_t row = 0; row < CNN_LINES_PAD; ++row) {
            conv_t* pprev_row = pprev + (row * CNN_COLS_PAD);
            conv_t* poutput_row = poutput + (row * CNN_COLS_PAD);
            *pprev_row = 0;
            *poutput_row = 0;
        }
    }

    conv_t bias_none = 0;
    conv_t* pbias_backup = (conv_t*)bias;
    conv_t* pbias = pbias_backup;

    CONV_loop_filter: for (i64_t f = 0; f < filters; ++f) {
        conv_t* pkernel = (conv_t*)kernel + (f * CHANNELS * C2D_KERNEL_LINES * C2D_KERNEL_COLS);

        conv_c_l_c output_offset;
        conv_t* poutput_offset;
        conv_t* pprev_offset;
        if (filters > 1) {
            output_offset = f * CNN_LINES_PAD * CNN_COLS_PAD;
            poutput_offset = output + output_offset;
            pprev_offset = prev + output_offset;
        }

        CONV_loop_channel: for (i64_t c = 0; c < CHANNELS; ++c) {
            conv_t* pkernel_c = pkernel + (c * C2D_KERNEL_LINES * C2D_KERNEL_COLS);
            conv_t* pinput_c = (conv_t*)input + (c * CNN_LINES_PAD * CNN_COLS_PAD);

            if (filters == 1) {
                output_offset = c * CNN_LINES_PAD * CNN_COLS_PAD;
                poutput_offset = output + output_offset;
                pprev_offset = prev + output_offset;
            }

            CONV_loop_row: for (conv_row_t orow = PADDING_OFFSET; orow < (CNN_LINES_PAD - PADDING_OFFSET); ++orow) {
                conv_t* poutput_offset_orow = poutput_offset + (orow * CNN_COLS_PAD);
                conv_t* pprev_offset_orow = pprev_offset + (orow * CNN_COLS_PAD);
                CONV_loop_col: for (conv_col_t ocol = PADDING_OFFSET; ocol < (inCols - PADDING_OFFSET); ++ocol) {
        #pragma HLS PIPELINE
                    conv_t acc = *pbias;
                    conv_t acc_sat;
                    conv_row_t korow = orow - PADDING_OFFSET;
                    CONV_loop_k1: for (conv_k_t krow = 0; krow < C2D_KERNEL_LINES; ++krow, ++korow) {
        //#pragma HLS PIPELINE
                        conv_t* pkernel_c_krow = pkernel_c + (krow * C2D_KERNEL_COLS);
                        conv_t* pinput_c_krow = pinput_c + (korow * CNN_COLS_PAD);
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
                    if (c == 0 || filters == 1) {
                        *pprev = 0;
                    }
                    acc_sat = acc + *pprev;
                    *poutput = acc_sat;
                    if ((c == 0 && filters == 1) || (filters > 1))
                        *pprev = acc_sat;
                    //printf("%9.6f ", acc_sat);
                }
                //printf("\n");
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


#endif // CONV2D_H
