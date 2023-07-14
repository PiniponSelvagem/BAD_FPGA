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
conv_t aux[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS];
conv_t* prev = (conv_t*)aux;

void conv2d(
    const i64_t filters,
    const i64_t channels,
    const conv_row_t inLines,
    //const conv_col_t inCols,
    const conv_row_t outLines,
    //const conv_col_t outCols,
    const conv_t* input,        // 3D array
    const conv_t* kernel,       // 3D array if filters==1, 4D array if filters>1
    const conv_t* bias,         // 1D array
    conv_t* output
) {

    // aux array: set usable bottom line to 0 because maxpool will reduce lines and old aux data is still there
    /*
    for (i64_t c = 0; c < CHANNELS; ++c) {
        for (conv_col_t col = 0; col < C2D_0__IN_COLS; ++col) {
            aux[c][C2D_0__IN_LINES-1][col] = 0;
        }
    }
    */

    conv_t bias_none = 0;
    conv_t* pbias_backup = (conv_t*)bias;
    conv_t* pbias = pbias_backup;

    CONV_loop_filter: for (i64_t f = 0; f < filters; ++f) {
        conv_t* pkernel = (conv_t*)kernel + (f * CHANNELS * C2D_KERNEL_LINES * C2D_KERNEL_COLS);

        conv_c_l_c output_offset;
        conv_t* poutput_offset;
        conv_t* pprev_offset;
        if (filters > 1) {
            output_offset = f * outLines * C2D_0__IN_COLS;
            poutput_offset = output + output_offset;
            pprev_offset = prev + output_offset;
        }

        CONV_loop_channel: for (i64_t c = 0; c < channels; ++c) {
            conv_t* pkernel_c = pkernel + (c * C2D_KERNEL_LINES * C2D_KERNEL_COLS);
            conv_t* pinput_c = (conv_t*)input + (c * inLines * C2D_0__IN_COLS);

            if (filters == 1) {
                output_offset = c * outLines * C2D_0__IN_COLS;
                poutput_offset = output + output_offset;
                pprev_offset = prev + output_offset;
            }

            CONV_loop_row: for (conv_row_t orow = PADDING_OFFSET; orow < (outLines - PADDING_OFFSET); ++orow) {
                conv_t* poutput_offset_orow = poutput_offset + (orow * C2D_0__IN_COLS);
                conv_t* pprev_offset_orow = pprev_offset + (orow * C2D_0__IN_COLS);
                CONV_loop_col: for (conv_col_t ocol = PADDING_OFFSET; ocol < (C2D_0__IN_COLS - PADDING_OFFSET); ++ocol) {
        #pragma HLS PIPELINE
                    conv_t acc = *pbias;
                    conv_t acc_sat;
                    conv_row_t korow = orow - PADDING_OFFSET;
                    CONV_loop_k1: for (conv_k_t krow = 0; krow < C2D_KERNEL_LINES; ++krow, ++korow) {
        //#pragma HLS PIPELINE
                        conv_t* pkernel_c_krow = pkernel_c + (krow * C2D_KERNEL_COLS);
                        conv_t* pinput_c_krow = pinput_c + (korow * C2D_0__IN_COLS);
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



template <int C_IN_LINES, int C_IN_COLS, int C_OUT_LINES, int C_OUT_COLS>
void conv2d_start(
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
                    conv_t k = kernel[krow][kcol];
                    conv_t i = input[korow][kocol];
                    acc += k * i;
/*
                    float bias_ = bias.to_float();
                    float korow_ = korow.to_float();
                    float kocol_ = kocol.to_float();
                    float k_ = k.to_float();
                    float i_ = i.to_float();
                    float acc_ = acc.to_float();

                    int abc = 0;
*/
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
            CONV_M_loop_k1: for (conv_k_t krow = 0; krow < C2D_KERNEL_LINES; ++krow, ++korow) {
//#pragma HLS PIPELINE
                conv_col_t kocol = ocol - PADDING_OFFSET;
            	CONV_M_loop_k2: for (conv_k_t kcol = 0; kcol < C2D_KERNEL_COLS; ++kcol, ++kocol) {
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
