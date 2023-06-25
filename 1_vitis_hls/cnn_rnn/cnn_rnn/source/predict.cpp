#include "input/input.h"

#include "conv2d/conv2d.h"
#include "bnorm/bnorm.h"

/* 0 */
#include "conv2d/data/conv2d_0.h"
#include "bnorm/data/bnorm_0.h"
#include "mpool2d/maxpool2d.h"

/* 1 */
#include "conv2d/data/conv2d_1.h"
#include "bnorm/data/bnorm_1.h"
#include "mpool2d/data/maxpool2d_0.h"


/* 0,1 */
input_t inputpad[C2D_0__IN_LINES][C2D_0__IN_COLS] = { 0 };
conv_t  outpad_01_a[CHANNELS][C2D_0__OUT_LINES][C2D_0__OUT_COLS] = { 0 };
conv_t  outpad_01_b[CHANNELS][C2D_0__OUT_LINES][C2D_0__OUT_COLS] = { 0 };

/* 2,3 */
//conv_t  outpad_23_a[CHANNELS][MP2D_0__OUT_LINES][MP2D_0__OUT_COLS] = { 0 };
conv_t  outpad_23_b[CHANNELS][MP2D_0__OUT_LINES][MP2D_0__OUT_COLS] = { 0 };


void predict(const input_t input[INPUT_LINES][INPUT_COLS], conv_t outpad_23_a[CHANNELS][MP2D_0__OUT_LINES][MP2D_0__OUT_COLS]) {
#pragma HLS INTERFACE s_axilite port=input bundle=BUS1
#pragma HLS INTERFACE s_axilite port=outpad_23_a bundle=BUS1
#pragma HLS INTERFACE s_axilite port=return bundle=BUS1

#pragma HLS ARRAY_PARTITION variable=inputpad type=cyclic factor=4 DIM=0
#pragma HLS ARRAY_PARTITION variable=outpad_01_b type=cyclic factor=4 DIM=0

    input_preconv2d(input, inputpad);

    /*
    TODO: change loop inter from "int" to "ap_uint"
    TODO: change loop inter from "int" to "ap_uint"
    TODO: change loop inter from "int" to "ap_uint"
    TODO: change loop inter from "int" to "ap_uint"
    */

    /*************************************/
    /**************** CNN ****************/
    /*************************************/
    /* 0 */
    // Conv2D
    P_CONV_0: for (int c = 0; c < CHANNELS; ++c) {
#pragma HLS PIPELINE off
        conv2d<C2D_0__IN_LINES, C2D_0__IN_COLS, C2D_0__OUT_LINES, C2D_0__OUT_COLS>(inputpad, kernel_0[c], bias_0[c], outpad_01_a[c]);
    }
    // BatchNormalization
    P_BNORM_0: for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_0__IN_LINES, BNORM_0__IN_COLS>(outpad_01_a[c], gamma_0[c], beta_0[c], movingmean_0[c], movingvariance_0[c], outpad_01_b[c]);
    }
    
    /* 1 */
    // Conv2D
    P_CONV_1: for (int f = 0; f < FILTERS; ++f) {
#pragma HLS PIPELINE off
        conv_t biasVal = bias_1[f];
        int c = 0;
        conv2d<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_01_b[c], kernel_1[c][f], biasVal, outpad_01_a[f]);
        ++c;
        P_CONV_1_M: for (; c < CHANNELS; ++c) {
#pragma HLS PIPELINE off
            conv2d_multi<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_01_b[c], outpad_01_a[f], kernel_1[c][f], outpad_01_a[f]);
        }
        break;
    }
    // BatchNormalization
    P_BNORM_1: for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_1__IN_LINES, BNORM_1__IN_COLS>(outpad_01_a[c], gamma_1[c], beta_1[c], movingmean_1[c], movingvariance_1[c], outpad_01_b[c]);
    }
    // MaxPool2D
    P_MAXPOOL_0: for (int c = 0; c < CHANNELS; ++c) {    /* 0 */
        maxpool2d<MP2D_0__IN_LINES, MP2D_0__IN_COLS, MP2D_0__OUT_LINES, MP2D_0__OUT_COLS>(outpad_01_b[c], outpad_23_a[c]);
    }
}








void test_input_preconv2d(const input_t input[INPUT_LINES][INPUT_COLS], conv_t inputpad[INPUT_PAD_LINES][INPUT_PAD_COLS]) {
#pragma HLS INTERFACE s_axilite port=input bundle=BUS1
#pragma HLS INTERFACE s_axilite port=inputpad bundle=BUS1
#pragma HLS INTERFACE s_axilite port=return bundle=BUS1

	input_preconv2d(input, inputpad);
}

void test_conv2d_0_c0(const input_t inputpad[INPUT_PAD_LINES][INPUT_PAD_COLS], conv_t output[CHANNELS][C2D_0__OUT_LINES][C2D_0__OUT_COLS]) {
#pragma HLS INTERFACE s_axilite port=inputpad bundle=BUS1
#pragma HLS INTERFACE s_axilite port=output bundle=BUS1
#pragma HLS INTERFACE s_axilite port=return bundle=BUS1

    conv2d<C2D_0__IN_LINES, C2D_0__IN_COLS, C2D_0__OUT_LINES, C2D_0__OUT_COLS>(inputpad, kernel_0[0], bias_0[0], output[0]);
}

void test_bnorm_0_c0(const input_t input[CHANNELS][C2D_0__OUT_LINES][C2D_0__OUT_COLS], conv_t output[CHANNELS][C2D_0__OUT_LINES][C2D_0__OUT_COLS]) {
#pragma HLS INTERFACE s_axilite port=input bundle=BUS1
#pragma HLS INTERFACE s_axilite port=output bundle=BUS1
#pragma HLS INTERFACE s_axilite port=return bundle=BUS1

    bnorm<BNORM_0__IN_LINES, BNORM_0__IN_COLS>(input[0], gamma_0[0], beta_0[0], movingmean_0[0], movingvariance_0[0], output[0]);
}
