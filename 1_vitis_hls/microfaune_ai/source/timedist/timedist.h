#pragma once

#ifndef TIMEDIST_H
#define TIMEDIST_H

#include "timedist_settings.h"
#ifdef __VITIS_HLS__
typedef ap_uint<9> tdist_row_t;
typedef ap_uint<8> tdist_col_t;
#endif
#ifdef _MSC_VER
typedef int tdist_row_t;
typedef int tdist_col_t;
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
        timedist_t* poutput_row = output + (row * kLines);
        TDIST_loop_col: for (tdist_col_t ocol = 0; ocol < TD_0__OUT_COLS; ++ocol) {
            if (ocol >= outCols)
                break;
            timedist_t acc = *(bias + ocol);
            timedist_t* pkernel_row = (timedist_t*)kernel + (ocol * kCols);
            timedist_t* pinput_row  = (timedist_t*)input + (row * inCols);
            TDIST_loop_kcol: for (tdist_col_t kcol = 0; kcol < TD_0__KERNEL_COLS; ++kcol) {
                if (kcol >= kCols)
                    break;
                timedist_t k = *(pkernel_row + kcol);
                timedist_t i = *(pinput_row + kcol);
                acc += k * i;
            }
            timedist_t* poutput = poutput_row + ocol;
            *poutput = SIGMOID(acc);
        }
    }
}

#endif // TIMEDIST_H
