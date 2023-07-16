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

// margin included
typedef ap_int<10> i512_t;  // requires signal because backward layer has check: i >= 0, and i will be -1


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
    conv2d(
        1/*FILTERS*/,
        C2D_0__IN_COLS,
        C2D_0__OUT_COLS,
        (conv_t*)inputpad,
        (conv_t*)kernel_0,
        (conv_t*)bias_0,
        (conv_t*)outpad_01_a
    );
    printf("CONV2D_0\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 8; ++c) {
            float a = outpad_01_a[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }
    // BatchNormalization
    bnorm(
        BNORM_0__IN_COLS,
        (bnorm_t*)outpad_01_a,
        (bnorm_t*)gamma_0,
        (bnorm_t*)beta_0,
        (bnorm_t*)movingmean_0,
        (bnorm_t*)movingvariance_0
    ); // BATCH NORMALIZATION + RELU
    printf("BNORM_0\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 8; ++c) {
            float a = outpad_01_a[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }

    /* 1 */
    // Conv2D
    conv2d(
        FILTERS,
        C2D_1__IN_COLS,
        C2D_1__OUT_COLS,
        (conv_t*)outpad_01_a,
        (conv_t*)kernel_1,
        (conv_t*)bias_1,
        (conv_t*)outpad_01_b
    );
    printf("CONV2D_1\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 8; ++c) {
            float a = outpad_01_b[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }
    // BatchNormalization
    bnorm(
        BNORM_1__IN_COLS,
        (bnorm_t*)outpad_01_b,
        (bnorm_t*)gamma_1,
        (bnorm_t*)beta_1,
        (bnorm_t*)movingmean_1,
        (bnorm_t*)movingvariance_1
    ); // BATCH NORMALIZATION + RELU
    printf("BNORM_1\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 8; ++c) {
            float a = outpad_01_b[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }
    // MaxPool2D
    maxpool2d(
        MP2D_0__IN_COLS,
        PADDING_OFFSET,
        (mpool_t*)outpad_01_b
    );
    printf("MAXPOOL_0\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 23; ++c) {
            float a = outpad_01_b[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }

    /* 2 */
    // Conv2D
    conv2d(
        FILTERS,
        C2D_2__IN_COLS,
        C2D_2__OUT_COLS,
        (conv_t*)outpad_01_b,
        (conv_t*)kernel_2,
        (conv_t*)bias_2,
        (conv_t*)outpad_01_a
    );
    printf("CONV2D_2\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 23; ++c) {
            float a = outpad_01_a[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }
    // BatchNormalization
    bnorm(
        BNORM_2__IN_COLS,
        (bnorm_t*)outpad_01_a,
        (bnorm_t*)gamma_2,
        (bnorm_t*)beta_2,
        (bnorm_t*)movingmean_2,
        (bnorm_t*)movingvariance_2
    ); // BATCH NORMALIZATION + RELU
    printf("BNORM_2\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 23; ++c) {
            float a = outpad_01_a[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }

    /* 3 */
    // Conv2D
    conv2d(
        FILTERS,
        C2D_3__IN_COLS,
        C2D_3__OUT_COLS,
        (conv_t*)outpad_01_a,
        (conv_t*)kernel_3,
        (conv_t*)bias_3,
        (conv_t*)outpad_01_b
    );
    printf("CONV2D_3\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 23; ++c) {
            float a = outpad_01_b[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }
    // BatchNormalization
    bnorm(
        BNORM_3__IN_COLS,
        (bnorm_t*)outpad_01_b,
        (bnorm_t*)gamma_3,
        (bnorm_t*)beta_3,
        (bnorm_t*)movingmean_3,
        (bnorm_t*)movingvariance_3
    ); // BATCH NORMALIZATION + RELU
    printf("BNORM_3\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 23; ++c) {
            float a = outpad_01_b[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }
    // MaxPool2D
    maxpool2d(
        MP2D_1__IN_COLS,
        PADDING_OFFSET,
        (mpool_t*)outpad_01_b
    );
    printf("MAXPOOL_1\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 23; ++c) {
            float a = outpad_01_b[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }

    /* 4 */
    // Conv2D
    // Conv2D
    conv2d(
        FILTERS,
        C2D_4__IN_COLS,
        C2D_4__OUT_COLS,
        (conv_t*)outpad_01_b,
        (conv_t*)kernel_4,
        (conv_t*)bias_4,
        (conv_t*)outpad_01_a
    );
    printf("CONV2D_4\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 12; ++c) {
            float a = outpad_01_a[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }
    // BatchNormalization
    bnorm(
        BNORM_4__IN_COLS,
        (bnorm_t*)outpad_01_a,
        (bnorm_t*)gamma_4,
        (bnorm_t*)beta_4,
        (bnorm_t*)movingmean_4,
        (bnorm_t*)movingvariance_4
    ); // BATCH NORMALIZATION + RELU
    printf("BNORM_4\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 12; ++c) {
            float a = outpad_01_a[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }

    /* 5 */
    // Conv2D
    conv2d(
        FILTERS,
        C2D_5__IN_COLS,
        C2D_5__OUT_COLS,
        (conv_t*)outpad_01_a,
        (conv_t*)kernel_5,
        (conv_t*)bias_5,
        (conv_t*)outpad_01_b
    );
    printf("CONV2D_5\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 12; ++c) {
            float a = outpad_01_b[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }
    // BatchNormalization
    bnorm(
        BNORM_5__IN_COLS,
        (bnorm_t*)outpad_01_b,
        (bnorm_t*)gamma_5,
        (bnorm_t*)beta_5,
        (bnorm_t*)movingmean_5,
        (bnorm_t*)movingvariance_5
    ); // BATCH NORMALIZATION + RELU
    printf("BNORM_5\n");
    for (int l = 0; l < 16; ++l) {
        for (int c = 0; c < 12; ++c) {
            float a = outpad_01_b[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }
    // MaxPool2D
    maxpool2d(
        MP2D_2__IN_COLS,
        0,
        (mpool_t*)outpad_01_b
    );
    printf("MAXPOOL_1\n");
    for (int l = 0; l < 431; ++l) {
        printf("%3d ", l);
        for (int c = 0; c < 5; ++c) {
            float a = outpad_01_b[63][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }

    /* 6 */
    // ReduceMax
    reducemax_0_saveTranspose(
        (reducemax_t*)outpad_01_b
    );
    printf("RMAX_0\n");
    for (int l = 0; l < RMAX_0__OUT_LINES; ++l) {
        printf("%3d ", l);
        for (int c = 0; c < RMAX_0__OUT_COLS; ++c) {
            reducemax_t (*poutput)[RMAX_0__OUT_COLS] = (reducemax_t(*)[64])outpad_01_b;
            float a = poutput[l][c];
            //float a = outpad_01_a[0][l][c];
            printf("%9.6f ", a);
        }
        printf("\n");
    }
}

void test(
    const input_t input[INPUT_LINES][INPUT_COLS],
    output_t outputLS[OUTPUT_LOCAL_SCORE_LINES][OUTPUT_LOCAL_SCORE_COLS],
    output_t outputGS[OUTPUT_GLOBAL_SCORE]
) {

    /*************************************/
    /**************** RNN ****************/
    /*************************************/
    /* 7 */
    // GRU 0 (forward)
    gru_clearState();
    P_GRU_0_F_LINES: for (i512_t i = 0; i < GRU_0__IN_LINES; ++i) {
        P_GRU_0_F_COLS: for (i64_t idx = 0; idx < GRU_0__IN_COLS; ++idx) {
            gru<GRU_0__IN_COLS, GRU_0__KERNEL_LINES, GRU_0__KERNEL_COLS, GRU_0__KERNEL_R_LINES, GRU_0__KERNEL_R_COLS, GRU_0__BIAS_SIZE>
                (idx, out_rmax0[i], kernel_gru0_f, bias_gru0_f, recurrent_kernel_gru0_f, recurrent_bias_gru0_f, &outgru_0[i][idx]);
        }
        gru_syncState();
    }
    // GRU 0 (backward)
    gru_clearState();
    P_GRU_0_B_LINES: for (i512_t i = GRU_0__IN_LINES-1; i >= 0; --i) {
        P_GRU_0_B_COLS: for (i64_t idx = 0; idx < GRU_0__IN_COLS; ++idx) {
            gru<GRU_0__IN_COLS, GRU_0__KERNEL_LINES, GRU_0__KERNEL_COLS, GRU_0__KERNEL_R_LINES, GRU_0__KERNEL_R_COLS, GRU_0__BIAS_SIZE>
                (idx, out_rmax0[i], kernel_gru0_b, bias_gru0_b, recurrent_kernel_gru0_b, recurrent_bias_gru0_b, &outgru_0[i][idx+64]);
        }
        gru_syncState();
    }

    /* 8 */
    // GRU 1 (forward)
    gru_clearState();
    P_GRU_1_F_LINES: for (i512_t i = 0; i < GRU_1__IN_LINES; ++i) {
        P_GRU_1_F_COLS: for (i128_t idx = 0; idx < GRU_1__IN_COLS; ++idx) {
            gru<GRU_1__IN_COLS, GRU_1__KERNEL_LINES, GRU_1__KERNEL_COLS, GRU_1__KERNEL_R_LINES, GRU_1__KERNEL_R_COLS, GRU_1__BIAS_SIZE>
                (idx, outgru_0[i], kernel_gru1_f, bias_gru1_f, recurrent_kernel_gru1_f, recurrent_bias_gru1_f, &outgru_1[i][idx]);
        }
        gru_syncState();
    }
    // GRU 1 (backward)
    gru_clearState();
    P_GRU_1_B_LINES: for (i512_t i = GRU_1__IN_LINES-1; i >= 0; --i) {
        P_GRU_1_B_COLS: for (i64_t idx = 0; idx < GRU_1__IN_COLS_BACK; ++idx) {
            gru<GRU_1__IN_COLS, GRU_1__KERNEL_LINES, GRU_1__KERNEL_COLS, GRU_1__KERNEL_R_LINES, GRU_1__KERNEL_R_COLS, GRU_1__BIAS_SIZE>
                (idx, outgru_0[i], kernel_gru1_b, bias_gru1_b, recurrent_kernel_gru1_b, recurrent_bias_gru1_b, &outgru_1[i][idx+64]);
        }
        gru_syncState();
    }

    /* 9 */
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