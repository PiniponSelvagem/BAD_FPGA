#include <stdio.h>

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

conv_t outarray_a[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS] = { 0 };
conv_t outarray_b[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS] = { 0 };


/* 0,1 */
input_t inputpad[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS] = { 0 };
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
    conv2d(  // TESTED AND VALIDATED
        1/*FILTERS*/,
        C2D_0__IN_COLS,
        (conv_t*)inputpad,
        (conv_t*)kernel_0,
        (conv_t*)bias_0,
        (conv_t*)outarray_a
    );
    for (int c = 0; c < CHANNELS; ++c) {
        conv2d_old<C2D_0__IN_LINES, C2D_0__IN_COLS, C2D_0__OUT_LINES, C2D_0__OUT_COLS>(inputpad[c], kernel_0[c], bias_0[c], outpad_01_a[c]);
    }
    // BatchNormalization
    bnorm(  // TESTED AND VALIDATED
        BNORM_0__IN_COLS,
        (bnorm_t*)outarray_a,
        (bnorm_t*)gamma_0,
        (bnorm_t*)beta_0,
        (bnorm_t*)movingmean_0,
        (bnorm_t*)movingvariance_0
    ); // BATCH NORMALIZATION + RELU
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm_old<BNORM_0__IN_LINES, BNORM_0__IN_COLS>(outpad_01_a[c], gamma_0[c], beta_0[c], movingmean_0[c], movingvariance_0[c], outpad_01_b[c]);
    }

    /* 1 */
    // Conv2D
    conv2d(  // TESTED AND VALIDATED
        FILTERS,
        C2D_1__IN_COLS,
        (conv_t*)outarray_a,
        (conv_t*)kernel_1,
        (conv_t*)bias_1,
        (conv_t*)outarray_b
    );
    for (int f = 0; f < FILTERS; ++f) {
        int c = 0;
        conv_t biasVal = bias_1[f];
        conv2d_old<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_01_b[c], kernel_1[f][c], biasVal, outpad_01_a[f]);
        ++c;
        for (; c < CHANNELS; ++c) {
            conv2d_multi<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_01_b[c], outpad_01_a[f], kernel_1[f][c], outpad_01_a[f]);
        }
    }
    // BatchNormalization
    bnorm(  // TESTED AND VALIDATED
        BNORM_1__IN_COLS,
        (bnorm_t*)outarray_b,
        (bnorm_t*)gamma_1,
        (bnorm_t*)beta_1,
        (bnorm_t*)movingmean_1,
        (bnorm_t*)movingvariance_1
    ); // BATCH NORMALIZATION + RELU
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm_old<BNORM_1__IN_LINES, BNORM_1__IN_COLS>(outpad_01_a[c], gamma_1[c], beta_1[c], movingmean_1[c], movingvariance_1[c], outpad_01_b[c]);
    }
    // MaxPool2D
    maxpool2d(
        MP2D_0__IN_COLS,
        MP2D_0__OUT_COLS,
        PADDING_OFFSET,
        (mpool_t*)outarray_b
    );
    for (int c = 0; c < CHANNELS; ++c) {    /* 0 */
        maxpool2d_old<MP2D_0__IN_LINES, MP2D_0__IN_COLS, MP2D_0__OUT_LINES, MP2D_0__OUT_COLS, PADDING_OFFSET>(outpad_01_b[c], outpad_23_a[c]);
    }

    /* 2 */
    // Conv2D
    conv2d(
        FILTERS,
        C2D_2__IN_COLS,
        (conv_t*)outarray_b,
        (conv_t*)kernel_2,
        (conv_t*)bias_2,
        (conv_t*)outarray_a
    );
    for (int f = 0; f < FILTERS; ++f) {
        int c = 0;
        conv_t biasVal = bias_2[f];
        conv2d_old<C2D_2__IN_LINES, C2D_2__IN_COLS, C2D_2__OUT_LINES, C2D_2__OUT_COLS>(outpad_23_a[c], kernel_2[f][c], biasVal, outpad_23_b[f]);
        ++c;
        for (; c < CHANNELS; ++c) {
            conv2d_multi<C2D_2__IN_LINES, C2D_2__IN_COLS, C2D_2__OUT_LINES, C2D_2__OUT_COLS>(outpad_23_a[c], outpad_23_b[f], kernel_2[f][c], outpad_23_b[f]);
        }
    }
    // BatchNormalization
    bnorm(
        BNORM_2__IN_COLS,
        (bnorm_t*)outarray_a,
        (bnorm_t*)gamma_2,
        (bnorm_t*)beta_2,
        (bnorm_t*)movingmean_2,
        (bnorm_t*)movingvariance_2
    ); // BATCH NORMALIZATION + RELU
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm_old<BNORM_2__IN_LINES, BNORM_2__IN_COLS>(outpad_23_b[c], gamma_2[c], beta_2[c], movingmean_2[c], movingvariance_2[c], outpad_23_a[c]);
    }
    

    /* 3 */
    // Conv2D
    conv2d(
        FILTERS,
        C2D_3__IN_COLS,
        (conv_t*)outarray_a,
        (conv_t*)kernel_3,
        (conv_t*)bias_3,
        (conv_t*)outarray_b
    );
    for (int f = 0; f < FILTERS; ++f) {
        int c = 0;
        conv_t biasVal = bias_3[f];
        conv2d_old<C2D_3__IN_LINES, C2D_3__IN_COLS, C2D_3__OUT_LINES, C2D_3__OUT_COLS>(outpad_23_a[c], kernel_3[f][c], biasVal, outpad_23_b[f]);
        ++c;
        for (; c < CHANNELS; ++c) {
            conv2d_multi<C2D_3__IN_LINES, C2D_3__IN_COLS, C2D_3__OUT_LINES, C2D_3__OUT_COLS>(outpad_23_a[c], outpad_23_b[f], kernel_3[f][c], outpad_23_b[f]);
        }
    }
    // BatchNormalization
    bnorm(
        BNORM_3__IN_COLS,
        (bnorm_t*)outarray_b,
        (bnorm_t*)gamma_3,
        (bnorm_t*)beta_3,
        (bnorm_t*)movingmean_3,
        (bnorm_t*)movingvariance_3
    ); // BATCH NORMALIZATION + RELU
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm_old<BNORM_3__IN_LINES, BNORM_3__IN_COLS>(outpad_23_b[c], gamma_3[c], beta_3[c], movingmean_3[c], movingvariance_3[c], outpad_23_a[c]);
    }
    // MaxPool2D
    maxpool2d(
        MP2D_1__IN_COLS,
        MP2D_1__OUT_COLS,
        PADDING_OFFSET,
        (mpool_t*)outarray_b
    );
    for (int c = 0; c < CHANNELS; ++c) {    /* 1 */
        maxpool2d_old<MP2D_1__IN_LINES, MP2D_1__IN_COLS, MP2D_1__OUT_LINES, MP2D_1__OUT_COLS, PADDING_OFFSET>(outpad_23_a[c], outpad_45_a[c]);
    }
    
    /* 4 */
    // Conv2D
    conv2d(
        FILTERS,
        C2D_4__IN_COLS,
        (conv_t*)outarray_b,
        (conv_t*)kernel_4,
        (conv_t*)bias_4,
        (conv_t*)outarray_a
    );
    for (int f = 0; f < FILTERS; ++f) {
        int c = 0;
        conv_t biasVal = bias_4[f];
        conv2d_old<C2D_4__IN_LINES, C2D_4__IN_COLS, C2D_4__OUT_LINES, C2D_4__OUT_COLS>(outpad_45_a[c], kernel_4[f][c], biasVal, outpad_45_b[f]);
        ++c;
        for (; c < CHANNELS; ++c) {
            conv2d_multi<C2D_4__IN_LINES, C2D_4__IN_COLS, C2D_4__OUT_LINES, C2D_4__OUT_COLS>(outpad_45_a[c], outpad_45_b[f], kernel_4[f][c], outpad_45_b[f]);
        }
    }
    // BatchNormalization
    bnorm(
        BNORM_4__IN_COLS,
        (bnorm_t*)outarray_a,
        (bnorm_t*)gamma_4,
        (bnorm_t*)beta_4,
        (bnorm_t*)movingmean_4,
        (bnorm_t*)movingvariance_4
    ); // BATCH NORMALIZATION + RELU
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm_old<BNORM_4__IN_LINES, BNORM_4__IN_COLS>(outpad_45_b[c], gamma_4[c], beta_4[c], movingmean_4[c], movingvariance_4[c], outpad_45_a[c]);
    }

    /* 5 */
    // Conv2D
    conv2d(
        FILTERS,
        C2D_5__IN_COLS,
        (conv_t*)outarray_a,
        (conv_t*)kernel_5,
        (conv_t*)bias_5,
        (conv_t*)outarray_b
    );
    for (int f = 0; f < FILTERS; ++f) {
        int c = 0;
        conv_t biasVal = bias_5[f];
        conv2d_old<C2D_5__IN_LINES, C2D_5__IN_COLS, C2D_5__OUT_LINES, C2D_5__OUT_COLS>(outpad_45_a[c], kernel_5[f][c], biasVal, outpad_45_b[f]);
        ++c;
        for (; c < CHANNELS; ++c) {
            conv2d_multi<C2D_5__IN_LINES, C2D_5__IN_COLS, C2D_5__OUT_LINES, C2D_5__OUT_COLS>(outpad_45_a[c], outpad_45_b[f], kernel_5[f][c], outpad_45_b[f]);
        }
    }
    // BatchNormalization
    bnorm(
        BNORM_5__IN_COLS,
        (bnorm_t*)outarray_b,
        (bnorm_t*)gamma_5,
        (bnorm_t*)beta_5,
        (bnorm_t*)movingmean_5,
        (bnorm_t*)movingvariance_5
    ); // BATCH NORMALIZATION + RELU
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm_old<BNORM_5__IN_LINES, BNORM_5__IN_COLS>(outpad_45_b[c], gamma_5[c], beta_5[c], movingmean_5[c], movingvariance_5[c], outpad_45_a[c]);
    }
    // MaxPool2D
    maxpool2d(
        MP2D_2__IN_COLS,
        MP2D_2__OUT_COLS,
        0,
        (mpool_t*)outarray_b
    );
    for (int c = 0; c < CHANNELS; ++c) {    /* 2 */
        maxpool2d_old<MP2D_2__IN_LINES, MP2D_2__IN_COLS, MP2D_2__OUT_LINES, MP2D_2__OUT_COLS, 0>(outpad_45_a[c], outpad_45_nopad[c]);
    }
    float(*arrx)[431][5] = (float(*)[431][5])outarray_b;
    for (int c = 0; c < 64; ++c) {
        for (int line = 0; line < 431; ++line) {
            for (int col = 0; col < 5; ++col) {
                float a = arrx[c][line][col];
                float b = outpad_45_nopad[c][line][col];
                if (a != b) {
                    printf("[%d][%d][%d]: %f != %f\n", c, line, col, a, b);
                }
            }
        }
    }
    printf("\n");

    /* 6 */
    // ReduceMax
    reducemax_0_saveTranspose(
        (reducemax_t*)outarray_b,
        (reducemax_t*)outarray_a
    );
    reducemax_0_saveTranspose_old<RMAX_0__IN_LINES, RMAX_0__IN_COLS, RMAX_0__OUT_LINES, RMAX_0__OUT_COLS>(outpad_45_nopad, out_rmax0);
    float(*arr)[64] = (float(*)[64])outarray_a;
    for (int line = 0; line < 431; ++line) {
        for (int col = 0; col < 64; ++col) {
            float a = arr[line][col];
            float b = out_rmax0[line][col];
            if (a != b) {
                printf("[%d][%d]: %f != %f\n", line, col, a, b);
            }
        }
    }
    printf("\n");
}

void rnn() {
    /*************************************/
    /**************** RNN ****************/
    /*************************************/
    /*
    // GRU_0 (CLEAR WITH PATTERN)
    for (int l = 0; l < GRU_0__OUT_LINES; ++l) {
        for (int c = 0; c < GRU_0__OUT_COLS; ++c) {
            reducemax_t (*poutput)[GRU_0__OUT_COLS] = (reducemax_t(*)[GRU_0__OUT_COLS])outpad_01_a;
            poutput[l][c] = 99.9999;
            float a = poutput[l][c];
        }
    }
    /* 7 *
    // GRU 0 (forward)
    gru(
        GRU_FORWARD,
        GRU_0__IN_COLS,
        GRU_0__KERNEL_LINES,
        (gru_t*)outpad_01_b,
        (gru_t*)kernel_gru0_f,           (gru_t*)bias_gru0_f,
        (gru_t*)recurrent_kernel_gru0_f, (gru_t*)recurrent_bias_gru0_f,
        (gru_t*)outpad_01_a
    );
    printf("GRU_0 (forward)\n");
    for (int l = 0; l < 16; ++l) {
        printf("%3d ", l);
        gru_t(*poutput)[GRU_0__OUT_COLS] = (gru_t(*)[GRU_0__OUT_COLS])outpad_01_a;
        for (int c = 0; c < 4; ++c) {
            float a = poutput[l][c];
            //float a = outpad_01_a[0][l][c];
            if (a < 99)
                printf("%9.6f ", a);
            else
                printf("--.------ ");
        }
        for (int c = 64; c < 4+64; ++c) {
            float a = poutput[l][c];
            //float a = outpad_01_a[0][l][c];
            if (a < 99)
                printf("%9.6f ", a);
            else
                printf("--.------ ");
        }
        printf("\n");
    }
    // GRU 0 (backward)
    gru_t(*p_kernel)[GRU_KERNEL_COLS][GRU_SPLIT] = (gru_t(*)[GRU_KERNEL_COLS][GRU_SPLIT])kernel_gru0_b;
    gru(
        GRU_BACKWARD,
        GRU_0__IN_COLS,
        GRU_0__KERNEL_LINES,
        (gru_t*)outpad_01_b,
        (gru_t*)kernel_gru0_b,           (gru_t*)bias_gru0_b,               // TODO: for some reason kernel_gru0_b pointer is equal to kernel_gru0_f, check this
        (gru_t*)recurrent_kernel_gru0_b, (gru_t*)recurrent_bias_gru0_b,
        (gru_t*)outpad_01_a
    );
    printf("GRU_0 (backward)\n");
    for (int l = 0; l < 16; ++l) {
        printf("%3d ", l);
        gru_t(*poutput)[GRU_0__OUT_COLS] = (gru_t(*)[GRU_0__OUT_COLS])outpad_01_a;
        for (int c = 0; c < 4; ++c) {
            float a = poutput[l][c];
            //float a = outpad_01_a[0][l][c];
            if (a < 99)
                printf("%9.6f ", a);
            else
                printf("--.------ ");
        }
        for (int c = 64; c < 4+64; ++c) {
            float a = poutput[l][c];
            //float a = outpad_01_a[0][l][c];
            if (a < 99)
                printf("%9.6f ", a);
            else
                printf("--.------ ");
        }
        printf("\n");
    }

    // GRU_1 (CLEAR WITH PATTERN)
    for (int l = 0; l < GRU_0__OUT_LINES; ++l) {
        for (int c = 0; c < GRU_0__OUT_COLS; ++c) {
            reducemax_t (*poutput)[GRU_0__OUT_COLS] = (reducemax_t(*)[GRU_0__OUT_COLS])outpad_01_b;
            poutput[l][c] = 99.9999;
            float a = poutput[l][c];
        }
    }
    /* 8 *
    // GRU 1 (forward)
    gru(
        GRU_FORWARD,
        GRU_1__IN_COLS,
        GRU_1__KERNEL_LINES,
        (gru_t*)outpad_01_a,
        (gru_t*)kernel_gru1_f,           (gru_t*)bias_gru1_f,
        (gru_t*)recurrent_kernel_gru1_f, (gru_t*)recurrent_bias_gru1_f,
        (gru_t*)outpad_01_b
    );
    printf("GRU_1 (forward)\n");
    for (int l = 0; l < 16; ++l) {
        printf("%3d ", l);
        reducemax_t (*poutput)[GRU_1__OUT_COLS] = (reducemax_t(*)[GRU_1__OUT_COLS])outpad_01_b;
        for (int c = 0; c < 4; ++c) {
            float a = poutput[l][c];
            //float a = outpad_01_a[0][l][c];
            if (a < 99)
                printf("%9.6f ", a);
            else
                printf("--.------ ");
        }
        for (int c = 64; c < 4 + 64; ++c) {
            float a = poutput[l][c];
            //float a = outpad_01_a[0][l][c];
            if (a < 99)
                printf("%9.6f ", a);
            else
                printf("--.------ ");
        }
        printf("\n");
    }
    // GRU 1 (backward)
    gru(
        GRU_BACKWARD,
        GRU_1__IN_COLS,
        GRU_1__KERNEL_LINES,
        (gru_t*)outpad_01_a,
        (gru_t*)kernel_gru1_b,           (gru_t*)bias_gru1_b,
        (gru_t*)recurrent_kernel_gru1_b, (gru_t*)recurrent_bias_gru1_b,
        (gru_t*)outpad_01_b
    );
    printf("GRU_1 (backward)\n");
    for (int l = 0; l < 16; ++l) {
        printf("%3d ", l);
        reducemax_t (*poutput)[GRU_1__OUT_COLS] = (reducemax_t(*)[GRU_1__OUT_COLS])outpad_01_b;
        for (int c = 0; c < 4; ++c) {
            float a = poutput[l][c];
            //float a = outpad_01_a[0][l][c];
            if (a < 99)
                printf("%9.6f ", a);
            else
                printf("--.------ ");
        }
        for (int c = 64; c < 4 + 64; ++c) {
            float a = poutput[l][c];
            //float a = outpad_01_a[0][l][c];
            if (a < 99)
                printf("%9.6f ", a);
            else
                printf("--.------ ");
        }
        printf("\n");
    }

    /* 9 *
    // TimeDistribution 0 ( + Dense)
    timedistributed_dense(
        TD_0__IN_COLS,
        TD_0__KERNEL_LINES, TD_0__KERNEL_COLS,
        TD_0__BIAS_SIZE,
        TD_0__OUT_COLS,
        (timedist_t*)outpad_01_b,
        (timedist_t*)kernel_td0,
        (timedist_t*)bias_td0,
        (timedist_t*)outpad_01_a
    );

    // TimeDistribution 1 ( + Dense)
    timedistributed_dense(
        TD_1__IN_COLS,
        TD_1__KERNEL_LINES, TD_0__KERNEL_COLS,
        TD_1__BIAS_SIZE,
        TD_1__OUT_COLS,
        (timedist_t*)outpad_01_a,
        (timedist_t*)kernel_td1,
        (timedist_t*)bias_td1,
        (timedist_t*)outpad_01_b
    );
    printf("\n");
    */
}

void test(
    const input_t input[INPUT_LINES][INPUT_COLS],
    output_t outputLS[OUTPUT_LOCAL_SCORE_LINES][OUTPUT_LOCAL_SCORE_COLS],
    output_t outputGS[OUTPUT_GLOBAL_SCORE]
) {
    /* 9 *
    // TimeDistribution 0 (Dense)
    P_TDIST_0: for (i512_t i = 0; i < INPUT_LINES; ++i) {
        timedistributed_dense<TD_0__IN_LINES, TD_0__IN_COLS, TD_0__KERNEL_LINES, TD_0__KERNEL_COLS, TD_0__BIAS_SIZE, TD_0__OUT_LINES, TD_0__OUT_COLS>
            (outgru_1[i], kernel_td0, bias_td0, outtd_0[i]);
    }
    // TimeDistribution 1 (Dense)
    P_TDIST_1: for (i512_t i = 0; i < INPUT_LINES; ++i) {
#pragma HLS PIPELINE off
        timedistributed_dense<TD_1__IN_LINES, TD_1__IN_COLS, TD_1__KERNEL_LINES, TD_1__KERNEL_COLS, TD_1__BIAS_SIZE, TD_1__OUT_LINES, TD_1__OUT_COLS>
            (outtd_0[i], kernel_td1, bias_td1, outputLS[i]);
    }
    
    /* 10 *
    reducemax_1<RMAX_1__IN_LINES>(*outputLS, outputGS);
    */
}

