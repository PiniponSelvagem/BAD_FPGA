#include "types.h"

#include "input/input.h"
#include "output/output.h"

#include "conv2d/conv2d.h"
#include "bnorm/bnorm.h"
#include "mpool2d/maxpool2d.h"
#include "reducemax/reducemax.h"
#include "gru/gru.h"
#include "timedist/timedist.h"

/* 0 */
#include "conv2d/data/conv2d_0.h"
#include "bnorm/data/bnorm_0.h"

/* 1 */
#include "conv2d/data/conv2d_1.h"
#include "bnorm/data/bnorm_1.h"
#include "mpool2d/data/maxpool2d_0.h"

/* 2 */
#include "conv2d/data/conv2d_2.h"
#include "bnorm/data/bnorm_2.h"

/* 3 */
#include "conv2d/data/conv2d_3.h"
#include "bnorm/data/bnorm_3.h"
#include "mpool2d/data/maxpool2d_1.h"

/* 4 */
#include "conv2d/data/conv2d_4.h"
#include "bnorm/data/bnorm_4.h"

/* 5 */
#include "conv2d/data/conv2d_5.h"
#include "bnorm/data/bnorm_5.h"
#include "mpool2d/data/maxpool2d_2.h"

/* 6 */
#include "reducemax/data/reducemax_0.h"

/* 7 */
#include "gru/data/gru_0_forward.h"
#include "gru/data/gru_0_backward.h"

/* 8 */
#include "gru/data/gru_1_forward.h"
#include "gru/data/gru_1_backward.h"

/* 9 */
#include "timedist/data/timedist_0.h"
#include "timedist/data/timedist_1.h"

/* 10 */
#include "reducemax/data/reducemax_1.h"


// SHOULD ONLY BE USED, in HLS to load the weights in test_bench
#include "load_weights.h"


/* 0,1 */
input_t inputpad[C2D_0__IN_LINES][C2D_0__IN_COLS] = { 0 };
conv_t  outpad_01_a[CHANNELS][C2D_0__OUT_LINES][C2D_0__OUT_COLS] = { 0 };
conv_t  outpad_01_b[CHANNELS][C2D_0__OUT_LINES][C2D_0__OUT_COLS] = { 0 };

/* 2,3 */
conv_t  outpad_23_a[CHANNELS][MP2D_0__OUT_LINES][MP2D_0__OUT_COLS] = { 0 };
conv_t  outpad_23_b[CHANNELS][MP2D_0__OUT_LINES][MP2D_0__OUT_COLS] = { 0 };

/* 4,5 */
conv_t  outpad_45_a[CHANNELS][MP2D_1__OUT_LINES][MP2D_1__OUT_COLS] = { 0 };
conv_t  outpad_45_b[CHANNELS][MP2D_1__OUT_LINES][MP2D_1__OUT_COLS] = { 0 };
// transition array with removed padding
mpool_t outpad_45_nopad[CHANNELS][MP2D_2__OUT_LINES][MP2D_2__OUT_COLS] = { 0 };

/* 6 */
reducemax_t out_rmax0[RMAX_0__OUT_LINES][RMAX_0__OUT_COLS] = { 0 };

/* 7 */
gru_t outgru_0[GRU_0__OUT_LINES][GRU_0__OUT_COLS] = { 0.0 };

/* 8 */
gru_t outgru_1[GRU_1__OUT_LINES][GRU_1__OUT_COLS] = { 0.0 };

/* 9 */
output_t outtd_0[INPUT_LINES][TD_0__OUT_COLS] = { 0.0 };
output_t outtd_1[INPUT_LINES][TD_1__OUT_COLS] = { 0.0 };

/* 10 */
// outputLS
// outputGS



// margin included
typedef ap_uint<7> i64_t;
typedef ap_uint<8> i128_t;
typedef ap_int<10> i431_t;  // requires signal because backward layer has check: i >= 0, and i will be -1

void predict(
    const input_t input[INPUT_LINES][INPUT_COLS],
    output_t outputLS[OUTPUT_LOCAL_SCORE_LINES][OUTPUT_LOCAL_SCORE_COLS],
    output_t outputGS[OUTPUT_GLOBAL_SCORE]
) {
#pragma HLS INTERFACE s_axilite port=input bundle=BUS1
#pragma HLS INTERFACE s_axilite port=outputLS bundle=BUS1
#pragma HLS INTERFACE s_axilite port=outputGS bundle=BUS1
#pragma HLS INTERFACE s_axilite port=return bundle=BUS1

#pragma HLS ARRAY_PARTITION variable=inputpad type=cyclic factor=4 DIM=0
#pragma HLS ARRAY_PARTITION variable=outpad_01_b type=cyclic factor=4 DIM=0
#pragma HLS ARRAY_PARTITION variable=outpad_23_a type=cyclic factor=4 DIM=0
#pragma HLS ARRAY_PARTITION variable=outpad_45_a type=cyclic factor=4 DIM=0
#pragma HLS ARRAY_PARTITION variable=outpad_45_nopad type=cyclic factor=3 DIM=0
#pragma HLS ARRAY_PARTITION variable=outtd_0 type=cyclic factor=2 DIM=0

    input_preconv2d(input, inputpad);


    /*************************************/
    /**************** CNN ****************/
    /*************************************/
    /* 0 */
    // Conv2D
    P_CONV_0: for (i64_t c = 0; c < CHANNELS; ++c) {
#pragma HLS PIPELINE off
        conv2d<C2D_0__IN_LINES, C2D_0__IN_COLS, C2D_0__OUT_LINES, C2D_0__OUT_COLS>(inputpad, kernel_0[c], bias_0[c], outpad_01_a[c]);
    }
    // BatchNormalization
    P_BNORM_0: for (i64_t c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_0__IN_LINES, BNORM_0__IN_COLS>(outpad_01_a[c], gamma_0[c], beta_0[c], movingmean_0[c], movingvariance_0[c], outpad_01_b[c]);
    }
    
    /* 1 */
    // Conv2D
    P_CONV_1: for (i64_t f = 0; f < FILTERS; ++f) {
#pragma HLS PIPELINE off
        conv_t biasVal = bias_1[f];
        i64_t c = 0;
        conv2d<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_01_b[c], kernel_1[c][f], biasVal, outpad_01_a[f]);
        ++c;
        P_CONV_1_M: for (; c < CHANNELS; ++c) {
#pragma HLS PIPELINE off
            conv2d_multi<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_01_b[c], outpad_01_a[f], kernel_1[c][f], outpad_01_a[f]);
        }
    }
    // BatchNormalization
    P_BNORM_1: for (i64_t c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_1__IN_LINES, BNORM_1__IN_COLS>(outpad_01_a[c], gamma_1[c], beta_1[c], movingmean_1[c], movingvariance_1[c], outpad_01_b[c]);
    }
    // MaxPool2D
    P_MAXPOOL_0: for (i64_t c = 0; c < CHANNELS; ++c) {    /* 0 */
        maxpool2d<MP2D_0__IN_LINES, MP2D_0__IN_COLS, MP2D_0__OUT_LINES, MP2D_0__OUT_COLS, PADDING_OFFSET>(outpad_01_b[c], outpad_23_a[c]);
    }

     /* 2 */
    // Conv2D
    P_CONV_2: for (i64_t f = 0; f < FILTERS; ++f) {
#pragma HLS PIPELINE off
        conv_t biasVal = bias_2[f];
        i64_t c = 0;
        conv2d<C2D_2__IN_LINES, C2D_2__IN_COLS, C2D_2__OUT_LINES, C2D_2__OUT_COLS>(outpad_23_a[c], kernel_2[c][f], biasVal, outpad_23_b[f]);
        ++c;
        P_CONV_2_M: for (; c < CHANNELS; ++c) {
#pragma HLS PIPELINE off
            conv2d_multi<C2D_2__IN_LINES, C2D_2__IN_COLS, C2D_2__OUT_LINES, C2D_2__OUT_COLS>(outpad_23_a[c], outpad_23_b[f], kernel_2[c][f], outpad_23_b[f]);
        }
    }
    // BatchNormalization
    P_BNORM_2: for (i64_t c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_2__IN_LINES, BNORM_2__IN_COLS>(outpad_23_b[c], gamma_2[c], beta_2[c], movingmean_2[c], movingvariance_2[c], outpad_23_a[c]);
    }

    /* 3 */
    // Conv2D
    P_CONV_3: for (i64_t f = 0; f < FILTERS; ++f) {
#pragma HLS PIPELINE off
        conv_t biasVal = bias_3[f];
        i64_t c = 0;
        conv2d<C2D_3__IN_LINES, C2D_3__IN_COLS, C2D_3__OUT_LINES, C2D_3__OUT_COLS>(outpad_23_a[c], kernel_3[c][f], biasVal, outpad_23_b[f]);
        ++c;
        P_CONV_3_M: for (; c < CHANNELS; ++c) {
#pragma HLS PIPELINE off
            conv2d_multi<C2D_3__IN_LINES, C2D_3__IN_COLS, C2D_3__OUT_LINES, C2D_3__OUT_COLS>(outpad_23_a[c], outpad_23_b[f], kernel_3[c][f], outpad_23_b[f]);
        }
    }
    // BatchNormalization
    P_BNORM_3: for (i64_t c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_3__IN_LINES, BNORM_3__IN_COLS>(outpad_23_b[c], gamma_3[c], beta_3[c], movingmean_3[c], movingvariance_3[c], outpad_23_a[c]);
    }
    // MaxPool2D
    P_MAXPOOL_1: for (i64_t c = 0; c < CHANNELS; ++c) {    /* 1 */
        maxpool2d<MP2D_1__IN_LINES, MP2D_1__IN_COLS, MP2D_1__OUT_LINES, MP2D_1__OUT_COLS, PADDING_OFFSET>(outpad_23_a[c], outpad_45_a[c]);
    }

    /* 4 */
    // Conv2D
    P_CONV_4: for (i64_t f = 0; f < FILTERS; ++f) {
#pragma HLS PIPELINE off
        conv_t biasVal = bias_4[f];
        i64_t c = 0;
        conv2d<C2D_4__IN_LINES, C2D_4__IN_COLS, C2D_4__OUT_LINES, C2D_4__OUT_COLS>(outpad_45_a[c], kernel_4[c][f], biasVal, outpad_45_b[f]);
        ++c;
        P_CONV_4_M: for (; c < CHANNELS; ++c) {
#pragma HLS PIPELINE off
            conv2d_multi<C2D_4__IN_LINES, C2D_4__IN_COLS, C2D_4__OUT_LINES, C2D_4__OUT_COLS>(outpad_45_a[c], outpad_45_b[f], kernel_4[c][f], outpad_45_b[f]);
        }
    }
    // BatchNormalization
    P_BNORM_4: for (i64_t c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_4__IN_LINES, BNORM_4__IN_COLS>(outpad_45_b[c], gamma_4[c], beta_4[c], movingmean_4[c], movingvariance_4[c], outpad_45_a[c]);
    }

    /* 5 */
    // Conv2D
    P_CONV_5: for (i64_t f = 0; f < FILTERS; ++f) {
#pragma HLS PIPELINE off
        conv_t biasVal = bias_5[f];
        i64_t c = 0;
        conv2d<C2D_5__IN_LINES, C2D_5__IN_COLS, C2D_5__OUT_LINES, C2D_5__OUT_COLS>(outpad_45_a[c], kernel_5[c][f], biasVal, outpad_45_b[f]);
        ++c;
        P_CONV_5_M: for (; c < CHANNELS; ++c) {
#pragma HLS PIPELINE off
            conv2d_multi<C2D_5__IN_LINES, C2D_5__IN_COLS, C2D_5__OUT_LINES, C2D_5__OUT_COLS>(outpad_45_a[c], outpad_45_b[f], kernel_5[c][f], outpad_45_b[f]);
        }
    }
    // BatchNormalization
    P_BNORM_5: for (i64_t c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_5__IN_LINES, BNORM_5__IN_COLS>(outpad_45_b[c], gamma_5[c], beta_5[c], movingmean_5[c], movingvariance_5[c], outpad_45_a[c]);
    }
    // MaxPool2D
    P_MAXPOOL_2: for (i64_t c = 0; c < CHANNELS; ++c) {    /* 2 */
        maxpool2d<MP2D_2__IN_LINES, MP2D_2__IN_COLS, MP2D_2__OUT_LINES, MP2D_2__OUT_COLS, 0>(outpad_45_a[c], outpad_45_nopad[c]);
    }

    /* 6 */
    // ReduceMax
    reducemax_0_saveTranspose<RMAX_0__IN_LINES, RMAX_0__IN_COLS, RMAX_0__OUT_LINES, RMAX_0__OUT_COLS>(outpad_45_nopad, out_rmax0);


    /*************************************/
    /**************** RNN ****************/
    /*************************************/
    /* 7 */
    // GRU 0 (forward)
    gru_clearState();
    P_GRU_0_F_LINES: for (i431_t i = 0; i < GRU_0__IN_LINES; ++i) {
        P_GRU_0_F_COLS: for (i64_t idx = 0; idx < GRU_0__IN_COLS; ++idx) {
            gru<GRU_0__IN_COLS, GRU_0__KERNEL_LINES, GRU_0__KERNEL_COLS, GRU_0__KERNEL_R_LINES, GRU_0__KERNEL_R_COLS, GRU_0__BIAS_SIZE>
                (idx, out_rmax0[i], kernel_gru0_f, bias_gru0_f, recurrent_kernel_gru0_f, recurrent_bias_gru0_f, &outgru_0[i][idx]);
        }
        gru_syncState();
    }
    // GRU 0 (backward)
    gru_clearState();
    P_GRU_0_B_LINES: for (i431_t i = GRU_0__IN_LINES-1; i >= 0; --i) {
        P_GRU_0_B_COLS: for (i64_t idx = 0; idx < GRU_0__IN_COLS; ++idx) {
            gru<GRU_0__IN_COLS, GRU_0__KERNEL_LINES, GRU_0__KERNEL_COLS, GRU_0__KERNEL_R_LINES, GRU_0__KERNEL_R_COLS, GRU_0__BIAS_SIZE>
                (idx, out_rmax0[i], kernel_gru0_b, bias_gru0_b, recurrent_kernel_gru0_b, recurrent_bias_gru0_b, &outgru_0[i][idx+64]);
        }
        gru_syncState();
    }

    /* 8 */
    // GRU 1 (forward)
    gru_clearState();
    P_GRU_1_F_LINES: for (i431_t i = 0; i < GRU_1__IN_LINES; ++i) {
        P_GRU_1_F_COLS: for (i128_t idx = 0; idx < GRU_1__IN_COLS; ++idx) {
            gru<GRU_1__IN_COLS, GRU_1__KERNEL_LINES, GRU_1__KERNEL_COLS, GRU_1__KERNEL_R_LINES, GRU_1__KERNEL_R_COLS, GRU_1__BIAS_SIZE>
                (idx, outgru_0[i], kernel_gru1_f, bias_gru1_f, recurrent_kernel_gru1_f, recurrent_bias_gru1_f, &outgru_1[i][idx]);
        }
        gru_syncState();
    }
    // GRU 1 (backward)
    gru_clearState();
    P_GRU_1_B_LINES: for (i431_t i = GRU_1__IN_LINES-1; i >= 0; --i) {
        P_GRU_1_B_COLS: for (i64_t idx = 0; idx < GRU_1__IN_COLS_BACK; ++idx) {
            gru<GRU_1__IN_COLS, GRU_1__KERNEL_LINES, GRU_1__KERNEL_COLS, GRU_1__KERNEL_R_LINES, GRU_1__KERNEL_R_COLS, GRU_1__BIAS_SIZE>
                (idx, outgru_0[i], kernel_gru1_b, bias_gru1_b, recurrent_kernel_gru1_b, recurrent_bias_gru1_b, &outgru_1[i][idx+64]);
        }
        gru_syncState();
    }

    /* 9 */
    // TimeDistribution 0 (Dense)
    P_TDIST_0: for (i431_t i = 0; i < INPUT_LINES; ++i) {
        timedistributed_dense<TD_0__IN_LINES, TD_0__IN_COLS, TD_0__KERNEL_LINES, TD_0__KERNEL_COLS, TD_0__BIAS_SIZE, TD_0__OUT_LINES, TD_0__OUT_COLS>
            (outgru_1[i], kernel_td0, bias_td0, outtd_0[i]);
    }
    // TimeDistribution 1 (Dense)
    P_TDIST_1: for (i431_t i = 0; i < INPUT_LINES; ++i) {
#pragma HLS PIPELINE off
        timedistributed_dense<TD_1__IN_LINES, TD_1__IN_COLS, TD_1__KERNEL_LINES, TD_1__KERNEL_COLS, TD_1__BIAS_SIZE, TD_1__OUT_LINES, TD_1__OUT_COLS>
            (outtd_0[i], kernel_td1, bias_td1, outputLS[i]);
    }
    
    /* 10 */
    reducemax_1<RMAX_1__IN_LINES>(*outputLS, outputGS);
}







/*
void test_input_preconv2d(const input_t input[INPUT_LINES][INPUT_COLS], input_t inputpad[INPUT_PAD_LINES][INPUT_PAD_COLS]) {
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
*/
