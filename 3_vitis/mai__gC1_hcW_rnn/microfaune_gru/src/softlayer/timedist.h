#ifndef TIMEDIST_H
#define TIMEDIST_H

#include "../global_settings.h"
#include "../utils/sigmoid.h"

#include "../weights/time_distributed_0_kernel.h"
#include "../weights/time_distributed_0_bias.h"
#include "../weights/time_distributed_1_kernel.h"
#include "../weights/time_distributed_1_bias.h"


void timedistributed_dense(
    int inCols,
    int kLines, int kCols,
    int outCols,
    float* input,
    const float* kernel,
    const float* bias,
	float* output
) {
    TDIST_loop_row: for (int row = 0; row < IHEIGHT; ++row) {
        int pinput_offset_row = (row * inCols);
        int poutput_offset_row = (row * kLines);
        TDIST_loop_col: for (int ocol = 0; ocol < FILTERS; ++ocol) {
            if (ocol >= outCols)
                break;
            float acc = *(bias + ocol);
            int pkernel_offset_row = (ocol * kCols);
            TDIST_loop_kcol: for (int kcol = 0; kcol < FILTERS; ++kcol) {
                if (kcol >= kCols)
                    break;
                float k = *((float*)kernel + pkernel_offset_row + kcol);
                float i = *((float*)input + pinput_offset_row + kcol);
                acc += k * i;
            }
            float out = SIGMOID_FLOAT(acc);
            float* poutput = output + poutput_offset_row + ocol;
            *poutput = out;
        }
    }
}

void timedist_0(float* input, float* output) {
    timedistributed_dense( // TDIST_0 + Dense
    	GRU_CELLS*2,
        FILTERS, GRU_CELLS*2,
		FILTERS,
		input,
		td0_kernel,
		td0_bias,
		output
    );
}
void timedist_1(float* input, float* output) {
    timedistributed_dense( // TDIST_1 + Dense
    	FILTERS,
        1, FILTERS,
        1,
		input,
		td1_kernel,
		td1_bias,
		output   // LOCAL OUTPUT
    );
}

#endif // TIMEDIST_H
