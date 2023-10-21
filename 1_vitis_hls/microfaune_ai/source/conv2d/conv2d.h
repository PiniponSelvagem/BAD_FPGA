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

typedef ap_uint<21> conv_p_aux;
typedef ap_uint<21> conv_p_input;
typedef ap_uint<16> conv_p_kernel;
typedef ap_uint<7>  conv_p_bias;
typedef ap_uint<21> conv_p_output;
#endif
#ifdef _MSC_VER
typedef int conv_row_t;
typedef int conv_col_t;
typedef int conv_k_t;

typedef int conv_c_l_c;

typedef int conv_p_aux;
typedef int conv_p_input;
typedef int conv_p_kernel;
typedef int conv_p_bias;
typedef int conv_p_output;
#endif


// aux array in filter calculations, must have the size of the max input ever provided
conv_t aux[CHANNELS][CNN_LINES_PAD][CNN_COLS_PAD];
conv_t* prev = (conv_t*)aux;

/**
 * @brief Converts a value to quantized 4 bits.
 * @param value: Value to be quantized.
 * @return Quantized value
 *
 * @note In HLS version, this should be removed / replaced using HLS.
*/
static inline conv_t valueQuant(conv_t value) {
#define OFFSET_QUANT    0.0625
#define STEP_SIZE_QUANT 0.125
#define MIN_QUANT       0
#define MAX_QUANT       MAX_RELU_VALUE
    value = value + OFFSET_QUANT;
    if (value <= MIN_QUANT)
        return MIN_QUANT;
    else if (value >= MAX_QUANT)
        return MAX_QUANT;
    int step = int((value + 1) / STEP_SIZE_QUANT);
    return (step * STEP_SIZE_QUANT) - 1.0;
}

void conv2d(
    const i64_t filters,
    const conv_col_t inoutCols,
    const conv_t* input,        // 3D array
    const conv_t* kernel,       // 3D array if filters==1, 4D array if filters>1
    const conv_t* bias,         // 1D array
    conv_t* output
) {
#pragma HLS INTERFACE s_axilite port=filters bundle=BUS1
#pragma HLS INTERFACE s_axilite port=inoutCols bundle=BUS1
#pragma HLS INTERFACE s_axilite port=input bundle=BUS1
#pragma HLS INTERFACE s_axilite port=kernel bundle=BUS1
#pragma HLS INTERFACE s_axilite port=bias bundle=BUS1
#pragma HLS INTERFACE s_axilite port=output bundle=BUS1
#pragma HLS INTERFACE s_axilite port=return bundle=BUS1

//#pragma HLS ARRAY_PARTITION variable=aux type=cyclic factor=2 DIM=0

    // clear all prev and output
    // TODO: improve so it only clears necessary parts of the arrays
    CONV_loop_clear_channel: for (i64_t c = 0; c < CHANNELS; ++c) {
//#pragma HLS PIPELINE ii=1     //TODO: activating this pragma, locks up the synthesis, requiring manual stopping
        CONV_loop_clear_row: for (conv_row_t row = 0; row < CNN_LINES_PAD; ++row) {
//#pragma HLS PIPELINE ii=1     //TODO: activating this pragma requires 'aux' partition
            CONV_loop_clear_col: for (conv_col_t col = 0; col < CNN_COLS_PAD; ++col) {
#pragma HLS PIPELINE ii=1
                conv_p_aux p_counter = (c * CNN_LINES_PAD * CNN_COLS_PAD) + (row * CNN_COLS_PAD) + col;
                conv_t* pprev = (conv_t*)aux + p_counter;
                conv_t* poutput = output + p_counter;
                *pprev = 0;
                *poutput = 0;
            }
        }
    }


    conv_t currBias = TC(*bias);
    i64_t bias_counter = 0;

    CONV_loop_filter: for (i64_t f = 0; f < filters; ++f) {
//#pragma HLS PIPELINE ii=1     //TODO: activating this pragma, locks up the synthesis, requiring manual stopping
//WARNING: [HLS 200-960] Cannot flatten loop 'CONV_loop_filter' ... the outer loop is not a perfect loop.
        conv_c_l_c output_offset;
        conv_p_aux pprev_offset = (f * CNN_LINES_PAD * CNN_COLS_PAD);
        conv_p_output poutput_offset = (f * CNN_LINES_PAD * inoutCols);
        conv_p_kernel pkernel_offset_filter = (f * CHANNELS * C2D_KERNEL_LINES * C2D_KERNEL_COLS);
        CONV_loop_channel: for (i64_t c = 0; c < CHANNELS; ++c) {
//#pragma HLS PIPELINE ii=1     //TODO: activating this pragma, locks up the synthesis, requiring manual stopping
            conv_p_kernel pkernel_offset_channel = pkernel_offset_filter + (c * C2D_KERNEL_LINES * C2D_KERNEL_COLS);
            conv_p_input pinput_offset_channel = (c * CNN_LINES_PAD * inoutCols);
            if (filters == 1) {
                pprev_offset = (c * CNN_LINES_PAD * CNN_COLS_PAD);
                poutput_offset = (c * CNN_LINES_PAD * inoutCols);
            }
            CONV_loop_row: for (conv_row_t orow = PADDING_OFFSET; orow < (CNN_LINES_PAD - PADDING_OFFSET); ++orow) {
//#pragma HLS LOOP_FLATTEN off    // Vitis HLS crashes if this is not set to 'off', related to '*poutput' and '*pprev' write of variable 'acc_sat'
//#pragma HLS PIPELINE ii=1
//WARNING: [HLS 200-960] Cannot flatten loop 'CONV_loop_row' ... outer loop is not a perfect loop because there is nontrivial logic before entering the inner loop.
                conv_p_aux pprev_offset_orow = pprev_offset + (orow * CNN_COLS_PAD);
                conv_p_output poutput_offset_orow = poutput_offset + (orow * inoutCols);
                CONV_loop_col: for (conv_col_t ocol = PADDING_OFFSET; ocol < (CNN_COLS_PAD - PADDING_OFFSET); ++ocol) {
#pragma HLS PIPELINE ii=7
                    if (ocol >= (inoutCols - PADDING_OFFSET))
                        break;
                    conv_acc_t acc = currBias;
                    conv_acc_t acc_sat;
                    conv_row_t korow = orow - PADDING_OFFSET;
                    CONV_loop_k1: for (conv_k_t krow = 0; krow < C2D_KERNEL_LINES; ++krow, ++korow) {
//#pragma HLS PIPELINE ii=1     //WARNING: [HLS 214-189] Pipeline directive for loop 'CONV_loop_k1' ... because the loop is unrolled completely
                        conv_col_t kocol = ocol - PADDING_OFFSET;
                        conv_p_kernel pkernel_offset_row = pkernel_offset_channel + (krow * C2D_KERNEL_COLS);
                        conv_p_input pinput_offset_row = pinput_offset_channel + (korow * inoutCols);
                        CONV_loop_k2: for (conv_k_t kcol = 0; kcol < C2D_KERNEL_COLS; ++kcol, ++kocol) {
//#pragma HLS PIPELINE ii=1     //WARNING: [HLS 214-189] Pipeline directive for loop 'CONV_loop_k2' ... because the loop is unrolled completely
                            conv_p_kernel pkernel_offset_col = pkernel_offset_row + kcol;
                            conv_p_input pinput_offset_col = pinput_offset_row + kocol;
                            conv_t k = *(kernel + pkernel_offset_col); //conv_t k = kernel[f][c][krow][kcol];
                            conv_t i = *(input + pinput_offset_col);   //conv_t i = input[c][korow][kocol];
                            acc += TC(TC(k) * TC(i));
                        }
                    }
                    conv_t* poutput = (output + (poutput_offset_orow + ocol));
                    //conv_t* pprev = (conv_t*)prev + (pprev_offset_orow + ocol);
                    //acc_sat = TC(TC(acc) + TC(*poutput));
                    acc_sat = acc + (*poutput);
                #ifdef _MSC_VER
                    if (filters == 1 || c == CHANNELS - 1)
                        acc_sat = valueQuant(acc_sat);  // TODO: remove in HLS
                #endif

                    /* ReLu */
                    if (filters == 1 || c == CHANNELS - 1)
                        if (acc_sat > MAX_RELU_VALUE)
                            acc_sat = MAX_RELU_VALUE;
                        else if (acc_sat < 0)
                            acc_sat = 0;

                    /*
                    #ifndef USE_FLOAT
                    if (acc_sat > MAX_VALUE)
                        acc_sat = MAX_VALUE;
                    #endif
                    */
					#ifdef __VITIS_HLS__
						float acc_satF = acc_sat.to_float();
					#endif // __VITIS_HLS__
                    *poutput = acc_sat;
                    //if ((c == 0 && filters == 1) || (filters > 1))
                    //    *pprev = acc_sat;   // TODO: Memory dependency because of read for acc_sat, making pipeline not lower than ii=7
                }
            }
            if (filters == 1) {
                // Loop "channels" only advances bias if only 1 "filters"
                ++bias_counter;
                currBias = TC(*(bias + bias_counter));
            }
            else if (c >= 0) {
                // If "filters > 1", only 1st channel should have the bias value, others 0
                currBias = 0;
            }
        }

        // Set the currBias from 0 to next bias
        ++bias_counter;
        currBias = *(bias + bias_counter);
    }
}

#endif // CONV2D_H
