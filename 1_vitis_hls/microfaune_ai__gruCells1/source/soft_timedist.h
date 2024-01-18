#ifndef TIMEDIST_H
#define TIMEDIST_H

#include "size_bgru.h"
#include "utils.h"

#define TD_0__OUT_COLS		FILTERS
#define TD_0__KERNEL_COLS	FILTERS


void timedistributed_dense(
    int inCols,
    int kLines, int kCols,
    int outCols,
    float* input,
    float* kernel,
    float* bias,
	float* output
) {
    TDIST_loop_row: for (int row = 0; row < IHEIGHT; ++row) {
        int pinput_offset_row = (row * inCols);
        int poutput_offset_row = (row * kLines);
        TDIST_loop_col: for (int ocol = 0; ocol < TD_0__OUT_COLS; ++ocol) {
            if (ocol >= outCols)
                break;
            float acc = *(bias + ocol);
            int pkernel_offset_row = (ocol * kCols);
            TDIST_loop_kcol: for (int kcol = 0; kcol < TD_0__KERNEL_COLS; ++kcol) {
                if (kcol >= kCols)
                    break;
                float k = *((float*)kernel + pkernel_offset_row + kcol);
                float i = *((float*)input + pinput_offset_row + kcol);
                acc += k * i;
            }
            float out = SIGMOID(acc);
            float* poutput = output + poutput_offset_row + ocol;
            *poutput = out;
        }
    }
}

#endif // TIMEDIST_H
