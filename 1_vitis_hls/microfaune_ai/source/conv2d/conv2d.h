#pragma once

#ifndef CONV2D_H
#define CONV2D_H

#include <ap_int.h>
#include "conv2d_settings.h"
#include "data/conv2d_0.h"

typedef ap_uint<9> conv_row_t;
typedef ap_uint<6> conv_col_t;
typedef ap_uint<2> conv_k_t;

typedef ap_uint<17> conv_f_c_Kl_Kc;
typedef ap_uint<11> conv_c_Kl_Kc;
typedef ap_uint<22> conv_c_l_c;


// aux array in filter calculations, must have the size of the max input ever provided
conv_t aux[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS];
conv_t* prev = (conv_t*)aux;

void conv2d(
    const i64_t filters,
    const i64_t channels,
    const conv_row_t inLines,
    const conv_col_t inCols,
    const conv_row_t outLines,
    const conv_col_t outCols,
    const conv_t* input,        // 3D array
    const conv_t* kernel,       // 3D array if filters==1, 4D array if filters>1
    const conv_t* bias,         // 1D array
    conv_t* output
) {
    conv_t bias_none = 0;
    conv_t* pbias_backup = (conv_t*)bias;
    conv_t* pbias = pbias_backup;
    //conv_t (*input)[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS] = (conv_t (*)[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS]) pinput;
    conv_t(*kernelArr)[CHANNELS][C2D_KERNEL_LINES][C2D_KERNEL_COLS] = (conv_t(*)[CHANNELS][C2D_KERNEL_LINES][C2D_KERNEL_COLS]) kernel;
    //conv_t (*bias)[CHANNELS] = (conv_t (*)[CHANNELS]) pbias;
    //conv_t (*output)[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS] = (conv_t (*)[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS]) poutput;

    CONV_loop_filter: for (i64_t f = 0; f < filters; ++f) {
        conv_f_c_Kl_Kc kernel_offset_f = f * CHANNELS * C2D_KERNEL_LINES * C2D_KERNEL_COLS;

        conv_c_l_c output_offset;
        if (filters > 1) {
            output_offset = f * outLines * outCols;
        }

        //printf("\nf: %d ", f);
        CONV_loop_channel: for (i64_t c = 0; c < channels; ++c) {
            conv_c_Kl_Kc kernel_offset_c = c * C2D_KERNEL_LINES * C2D_KERNEL_COLS;
            conv_c_l_c input_offset_c  = c * inLines * inCols;

            if (filters == 1) {
                output_offset = c * outLines * outCols;
            }

            CONV_loop_row: for (conv_row_t orow = PADDING_OFFSET; orow < (outLines - PADDING_OFFSET); ++orow) {
                CONV_loop_col: for (conv_col_t ocol = PADDING_OFFSET; ocol < (outCols - PADDING_OFFSET); ++ocol) {
        #pragma HLS PIPELINE
                    conv_t acc = *pbias;
                    conv_t acc_sat;
                    conv_row_t korow = orow - PADDING_OFFSET;
                    CONV_loop_k1: for (conv_k_t krow = 0; krow < C2D_KERNEL_LINES; ++krow, ++korow) {
        //#pragma HLS PIPELINE
                        conv_col_t kocol = ocol - PADDING_OFFSET;
                        CONV_loop_k2: for (conv_k_t kcol = 0; kcol < C2D_KERNEL_COLS; ++kcol, ++kocol) {
        //#pragma HLS PIPELINE
// TODO: Improve pointer calculation
                            //k = kernelArr[f][c][krow][kcol];
                            conv_t k = *(kernel + (kernel_offset_f) + (kernel_offset_c) + (krow * C2D_KERNEL_COLS) + kcol);
                            //conv_t i = input[c][korow][kocol];
                            conv_t i = *(input + (input_offset_c + korow * inCols + kocol));
                            //if (c == 0 && korow == 1 && kocol == 1)
                            //   printf("%9.6f ", i);
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
                    conv_t* poutput = (output + (output_offset + orow * outCols + ocol));
                    conv_t* pprev = (prev + (output_offset + orow * inCols + ocol));
                    if (c == 0 || filters == 1) {
                        *pprev = 0;
                    }
                    acc_sat = acc + *pprev;
                    *poutput = acc_sat;
                    if (c == 0)
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
