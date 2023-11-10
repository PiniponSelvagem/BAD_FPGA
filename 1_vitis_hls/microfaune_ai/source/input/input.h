#pragma once

#ifndef INPUT_H
#define INPUT_H

#include "input_settings.h"

typedef ap_uint<9> inpreconv_h_t;
typedef ap_uint<6> inpreconv_w_t;

void input_preconv2d(const input_t input[INPUT_LINES][INPUT_COLS], input_t inputpad[CHANNELS][INPUT_PAD_LINES][INPUT_PAD_COLS]) {
//#pragma HLS ARRAY_PARTITION variable=input type=cyclic factor=2 DIM=0
//#pragma HLS ARRAY_PARTITION variable=input type=cyclic factor=4 DIM=1
//#pragma HLS ARRAY_PARTITION variable=inputpad type=cyclic factor=4 DIM=0

    for (i64_t c = 0; c < CHANNELS; ++c) {
        INPUT_loop_Lstart: for (inpreconv_w_t w = 0; w < (INPUT_PAD_COLS - 1); ++w) {
            inputpad[c][0][w] = 0;
        }

        INPUT_loop_H: for (inpreconv_h_t hP = 1, h = 0; hP < (INPUT_PAD_LINES - 1); ++hP, ++h) {
#pragma HLS PIPELINE off
            inputpad[c][hP][0] = 0;
            INPUT_loop_W: for (inpreconv_w_t wP = 1, w = 0; wP < (INPUT_PAD_COLS - 1); ++wP, ++w) {
#pragma HLS PIPELINE
                inputpad[c][hP][wP] = input[h][w];
            }
            inputpad[c][hP][(INPUT_PAD_COLS - 1)] = 0;
        }

        INPUT_loop_Lend: for (inpreconv_w_t w = 0; w < (INPUT_PAD_COLS - 1); ++w) {
            inputpad[c][(INPUT_PAD_LINES - 1)][w] = 0;
        }
    }
}

#endif // INPUT_H
