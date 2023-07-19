#pragma once

#ifndef TIMEDIST_H
#define TIMEDIST_H

#include "timedist_settings.h"
#ifdef __VITIS_HLS__
typedef ap_uint<8> tdist_row_t;
typedef ap_uint<7> tdist_col_t;
#endif
#ifdef _MSC_VER
typedef int tdist_row_t;
typedef int tdist_col_t;
#endif

void timedistributed_dense(
    i128_t inCols,
    i128_t kLines, i64_t kCols,
    i64_t biasSize,
    i64_t outCols,
    const timedist_t* input,
    const timedist_t* kernel,
    const timedist_t* bias,
    timedist_t* output
) {
    TDIST_loop_row: for (tdist_col_t row = 0; row < RNN_LINES_GRU; ++row) {
        TDIST_loop_col: for (tdist_col_t ocol = 0; ocol < TD_0__IN_COLS; ++ocol) {
            if (ocol >= outCols)
                break;
            timedist_t acc = *(bias + ocol); //bias[ocol];
            TDIST_loop_krow: for (tdist_row_t krow = 0; krow < TD_0__KERNEL_LINES; ++krow) {
                if (krow >= kLines)
                    break;
                timedist_t k = *(kernel + (krow * TD_0__KERNEL_LINES) + ocol); //kernel[krow][ocol];
                timedist_t i = *(input + (row * RNN_LINES_GRU) + krow); //input[i][krow];
                acc += k * i;
            }
            timedist_t* poutput = output + (row * RNN_LINES_GRU) + ocol;
            *poutput = SIGMOID(acc);
        }
    }
}



template <
    int TD_IN_LINES, int TD_IN_COLS,
    int TD_KERNEL_LINES, int TD_KERNEL_COLS,
    int TD_BIAS_SIZE,
    int TD_OUT_LINES, int TD_OUT_COLS
>
void timedistributed_dense_old(
    const timedist_t input[TD_IN_COLS],
    const timedist_t kernel[TD_KERNEL_LINES][TD_KERNEL_COLS],
    const timedist_t bias[TD_BIAS_SIZE],
    timedist_t output[TD_OUT_COLS]
) {
    TDIST_loop_col: for (tdist_col_t ocol = 0; ocol < TD_OUT_COLS; ++ocol) {
#pragma HLS PIPELINE off
        timedist_t acc = bias[ocol];
        TDIST_loop_row: for (tdist_row_t krow = 0; krow < TD_KERNEL_LINES; ++krow) {
            timedist_t k = kernel[krow][ocol];
            timedist_t i = input[krow];
            acc += k * i;
        }
        output[ocol] = SIGMOID(acc);
    }
}


#endif // TIMEDIST_H