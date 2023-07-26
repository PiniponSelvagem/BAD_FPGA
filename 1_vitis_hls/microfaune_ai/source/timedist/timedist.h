#pragma once

#ifndef TIMEDIST_H
#define TIMEDIST_H

#include "timedist_settings.h"
#ifdef __VITIS_HLS__
typedef ap_uint<9> tdist_row_t;
typedef ap_uint<8> tdist_col_t;

typedef ap_uint<16> tdist_p_input;
typedef ap_uint<14> tdist_p_kernel;
typedef ap_uint<15> tdist_p_output;
#endif
#ifdef _MSC_VER
typedef int tdist_row_t;
typedef int tdist_col_t;

typedef int tdist_p_input;
typedef int tdist_p_kernel;
typedef int tdist_p_output;
#endif

void timedistributed_dense(
    i128_t inCols,
    i64_t kLines, i128_t kCols,
    i64_t outCols,
    const timedist_t* input,
    const timedist_t* kernel,
    const timedist_t* bias,
    timedist_t* output
) {
    TDIST_loop_row: for (tdist_row_t row = 0; row < RNN_LINES_GRU; ++row) {
        tdist_p_input pinput_offset_row  = (row * inCols);
        tdist_p_output poutput_offset_row = (row * kLines);
        TDIST_loop_col: for (tdist_col_t ocol = 0; ocol < TD_0__OUT_COLS; ++ocol) {
            if (ocol >= outCols)
                break;
            timedist_t acc = *(bias + ocol);
            tdist_p_kernel pkernel_offset_row = (ocol * kCols);
            TDIST_loop_kcol: for (tdist_col_t kcol = 0; kcol < TD_0__KERNEL_COLS; ++kcol) {
                if (kcol >= kCols)
                    break;
                timedist_t k = *((timedist_t*)kernel + pkernel_offset_row + kcol);
                timedist_t i = *((timedist_t*)input + pinput_offset_row + kcol);
                acc += TC(TC(k) * TC(i));
            }
            timedist_t* poutput = output + poutput_offset_row + ocol;
            *poutput = TC(SIGMOID(TC(acc)));
        }
    }
}

#endif // TIMEDIST_H
