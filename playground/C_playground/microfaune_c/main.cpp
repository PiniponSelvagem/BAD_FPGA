#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>

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


#define USE_WFILES
#include "load_weights.h"



void special_print(float out, float difference, float exp) {
    char difference_str[50];
    // Use snprintf to format difference as a string with 4 digits after the decimal point
    snprintf(difference_str, sizeof(difference_str), "%.12f", difference);

    // Loop through the string and replace '0' characters with '_'
    for (int i = 0; i < strlen(difference_str); i++) {
        if (difference_str[i] == '0') {
            difference_str[i] = '_';
        }
    }

    // Print the formatted string with underscores
    printf(" %15.12f | %15s | %15.12f\n", out, difference_str, exp);
}


/* 0,1 */
float inputpad[C2D_0__IN_LINES][C2D_0__IN_COLS] = { 0 };
float outpad_01_a[CHANNELS][C2D_0__OUT_LINES][C2D_0__OUT_COLS] = { 0 };
float outpad_01_b[CHANNELS][C2D_0__OUT_LINES][C2D_0__OUT_COLS] = { 0 };

/* 2,3 */
float outpad_23_a[CHANNELS][MP2D_0__OUT_LINES][MP2D_0__OUT_COLS] = { 0 };
float outpad_23_b[CHANNELS][MP2D_0__OUT_LINES][MP2D_0__OUT_COLS] = { 0 };

/* 4,5 */
float outpad_45_a[CHANNELS][MP2D_1__OUT_LINES][MP2D_1__OUT_COLS] = { 0 };
float outpad_45_b[CHANNELS][MP2D_1__OUT_LINES][MP2D_1__OUT_COLS] = { 0 };
// transition array to remove padding
float outpad_45_nopad[CHANNELS][MP2D_2__OUT_LINES][MP2D_2__OUT_COLS] = { 0 };

/* 6 */
float out_rmax0[RMAX_0__OUT_LINES][RMAX_0__OUT_COLS] = { 0 };

/* 7 */
float outgru_0[GRU_0__OUT_LINES][GRU_0__OUT_COLS] = { 0.0 };

/* 8 */
float outgru_1[GRU_1__OUT_LINES][GRU_1__OUT_COLS] = { 0.0 };

/* 9 */
float outtd_0[INPUT_LINES][TD_0__OUT_COLS] = { 0.0 };
float outtd_1[INPUT_LINES][TD_1__OUT_COLS] = { 0.0 };

/* 10 */
// outputLS


// I know ifndef this way is a bit overkill, but removing "const" from the weights made compiler error "x is already defined".
// This happened even before adding load_weights.h.
void predict(
    const input_t input[INPUT_LINES][INPUT_COLS],
    output_t outputLS[OUTPUT_LOCAL_SCORE_LINES][OUTPUT_LOCAL_SCORE_COLS],
    output_t outputGS[OUTPUT_GLOBAL_SCORE])
{
    input_preconv2d(input, inputpad);

    /*************************************/
    /**************** CNN ****************/
    /*************************************/
    /* 0 */
    // Conv2D
    for (int c = 0; c < CHANNELS; ++c) {
    #ifndef USE_WFILES
        conv2d<C2D_0__IN_LINES, C2D_0__IN_COLS, C2D_0__OUT_LINES, C2D_0__OUT_COLS>(inputpad, kernel_0[c], bias_0[c], outpad_01_a[c]);
    #else
        conv2d<C2D_0__IN_LINES, C2D_0__IN_COLS, C2D_0__OUT_LINES, C2D_0__OUT_COLS>(inputpad, kernel_0_loaded[c], bias_0_loaded[c], outpad_01_a[c]);
    #endif
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
    #ifndef USE_WFILES
        bnorm<BNORM_0__IN_LINES, BNORM_0__IN_COLS>(outpad_01_a[c], gamma_0[c], beta_0[c], movingmean_0[c], movingvariance_0[c], outpad_01_b[c]);
    #else
        bnorm<BNORM_0__IN_LINES, BNORM_0__IN_COLS>(outpad_01_a[c], gamma_0_loaded[c], beta_0_loaded[c], movingmean_0_loaded[c], movingvariance_0_loaded[c], outpad_01_b[c]);
    #endif
    }

    /* 1 */
    // Conv2D
    for (int f = 0; f < FILTERS; ++f) {
        int c = 0;
    #ifndef USE_WFILES
        conv_t biasVal = bias_1[f];
        conv2d<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_01_b[c], kernel_1[f][c], biasVal, outpad_01_a[f]);
    #else
        conv_t biasVal = bias_1_loaded[f];
        conv2d<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_01_b[c], kernel_1_loaded[f][c], biasVal, outpad_01_a[f]);
    #endif
        ++c;
        for (; c < CHANNELS; ++c) {
        #ifndef USE_WFILES
            conv2d_multi<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_01_b[c], outpad_01_a[f], kernel_1[f][c], outpad_01_a[f]);
        #else
            conv2d_multi<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_01_b[c], outpad_01_a[f], kernel_1_loaded[f][c], outpad_01_a[f]);
        #endif
        }
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
    #ifndef USE_WFILES
        bnorm<BNORM_1__IN_LINES, BNORM_1__IN_COLS>(outpad_01_a[c], gamma_1[c], beta_1[c], movingmean_1[c], movingvariance_1[c], outpad_01_b[c]);
    #else
        bnorm<BNORM_1__IN_LINES, BNORM_1__IN_COLS>(outpad_01_a[c], gamma_1_loaded[c], beta_1_loaded[c], movingmean_1_loaded[c], movingvariance_1_loaded[c], outpad_01_b[c]);
    #endif
    }
    // MaxPool2D
    for (int c = 0; c < CHANNELS; ++c) {    /* 0 */
        maxpool2d<MP2D_0__IN_LINES, MP2D_0__IN_COLS, MP2D_0__OUT_LINES, MP2D_0__OUT_COLS, PADDING_OFFSET>(outpad_01_b[c], outpad_23_a[c]);
    }

    /* 2 */
    // Conv2D
    for (int f = 0; f < FILTERS; ++f) {
        int c = 0;
    #ifndef USE_WFILES
        conv_t biasVal = bias_2[f];
        conv2d<C2D_2__IN_LINES, C2D_2__IN_COLS, C2D_2__OUT_LINES, C2D_2__OUT_COLS>(outpad_23_a[c], kernel_2[f][c], biasVal, outpad_23_b[f]);
    #else
        conv_t biasVal = bias_2_loaded[f];
        conv2d<C2D_2__IN_LINES, C2D_2__IN_COLS, C2D_2__OUT_LINES, C2D_2__OUT_COLS>(outpad_23_a[c], kernel_2_loaded[f][c], biasVal, outpad_23_b[f]);
    #endif
        ++c;
        for (; c < CHANNELS; ++c) {
        #ifndef USE_WFILES
            conv2d_multi<C2D_2__IN_LINES, C2D_2__IN_COLS, C2D_2__OUT_LINES, C2D_2__OUT_COLS>(outpad_23_a[c], outpad_23_b[f], kernel_2[f][c], outpad_23_b[f]);
        #else
            conv2d_multi<C2D_2__IN_LINES, C2D_2__IN_COLS, C2D_2__OUT_LINES, C2D_2__OUT_COLS>(outpad_23_a[c], outpad_23_b[f], kernel_2_loaded[f][c], outpad_23_b[f]);
        #endif
        }
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
    #ifndef USE_WFILES
        bnorm<BNORM_2__IN_LINES, BNORM_2__IN_COLS>(outpad_23_b[c], gamma_2[c], beta_2[c], movingmean_2[c], movingvariance_2[c], outpad_23_a[c]);
    #else
        bnorm<BNORM_2__IN_LINES, BNORM_2__IN_COLS>(outpad_23_b[c], gamma_2_loaded[c], beta_2_loaded[c], movingmean_2_loaded[c], movingvariance_2_loaded[c], outpad_23_a[c]);
    #endif
    }

    /* 3 */
    // Conv2D
    for (int f = 0; f < FILTERS; ++f) {
        int c = 0;
    #ifndef USE_WFILES
        conv_t biasVal = bias_3[f];
        conv2d<C2D_3__IN_LINES, C2D_3__IN_COLS, C2D_3__OUT_LINES, C2D_3__OUT_COLS>(outpad_23_a[c], kernel_3[f][c], biasVal, outpad_23_b[f]);
    #else
        conv_t biasVal = bias_3_loaded[f];
        conv2d<C2D_3__IN_LINES, C2D_3__IN_COLS, C2D_3__OUT_LINES, C2D_3__OUT_COLS>(outpad_23_a[c], kernel_3_loaded[f][c], biasVal, outpad_23_b[f]);
    #endif
        ++c;
        for (; c < CHANNELS; ++c) {
        #ifndef USE_WFILES
            conv2d_multi<C2D_3__IN_LINES, C2D_3__IN_COLS, C2D_3__OUT_LINES, C2D_3__OUT_COLS>(outpad_23_a[c], outpad_23_b[f], kernel_3[f][c], outpad_23_b[f]);
        #else
            conv2d_multi<C2D_3__IN_LINES, C2D_3__IN_COLS, C2D_3__OUT_LINES, C2D_3__OUT_COLS>(outpad_23_a[c], outpad_23_b[f], kernel_3_loaded[f][c], outpad_23_b[f]);
        #endif
        }
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
    #ifndef USE_WFILES
        bnorm<BNORM_3__IN_LINES, BNORM_3__IN_COLS>(outpad_23_b[c], gamma_3[c], beta_3[c], movingmean_3[c], movingvariance_3[c], outpad_23_a[c]);
    #else
        bnorm<BNORM_3__IN_LINES, BNORM_3__IN_COLS>(outpad_23_b[c], gamma_3_loaded[c], beta_3_loaded[c], movingmean_3_loaded[c], movingvariance_3_loaded[c], outpad_23_a[c]);
    #endif
    }
    // MaxPool2D
    for (int c = 0; c < CHANNELS; ++c) {    /* 1 */
        maxpool2d<MP2D_1__IN_LINES, MP2D_1__IN_COLS, MP2D_1__OUT_LINES, MP2D_1__OUT_COLS, PADDING_OFFSET>(outpad_23_a[c], outpad_45_a[c]);
    }

    /* 4 */
    // Conv2D
    for (int f = 0; f < FILTERS; ++f) {
        int c = 0;
    #ifndef USE_WFILES
        conv_t biasVal = bias_4[f];
        conv2d<C2D_4__IN_LINES, C2D_4__IN_COLS, C2D_4__OUT_LINES, C2D_4__OUT_COLS>(outpad_45_a[c], kernel_4[f][c], biasVal, outpad_45_b[f]);
    #else
        conv_t biasVal = bias_4_loaded[f];
        conv2d<C2D_4__IN_LINES, C2D_4__IN_COLS, C2D_4__OUT_LINES, C2D_4__OUT_COLS>(outpad_45_a[c], kernel_4_loaded[f][c], biasVal, outpad_45_b[f]);
    #endif
        ++c;
        for (; c < CHANNELS; ++c) {
        #ifndef USE_WFILES
            conv2d_multi<C2D_4__IN_LINES, C2D_4__IN_COLS, C2D_4__OUT_LINES, C2D_4__OUT_COLS>(outpad_45_a[c], outpad_45_b[f], kernel_4[f][c], outpad_45_b[f]);
        #else
            conv2d_multi<C2D_4__IN_LINES, C2D_4__IN_COLS, C2D_4__OUT_LINES, C2D_4__OUT_COLS>(outpad_45_a[c], outpad_45_b[f], kernel_4_loaded[f][c], outpad_45_b[f]);
        #endif
        }
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
    #ifndef USE_WFILES
        bnorm<BNORM_4__IN_LINES, BNORM_4__IN_COLS>(outpad_45_b[c], gamma_4[c], beta_4[c], movingmean_4[c], movingvariance_4[c], outpad_45_a[c]);
    #else
        bnorm<BNORM_4__IN_LINES, BNORM_4__IN_COLS>(outpad_45_b[c], gamma_4_loaded[c], beta_4_loaded[c], movingmean_4_loaded[c], movingvariance_4_loaded[c], outpad_45_a[c]);
    #endif
    }

    /* 5 */
    // Conv2D
    for (int f = 0; f < FILTERS; ++f) {
        int c = 0;
    #ifndef USE_WFILES
        conv_t biasVal = bias_5[f];
        conv2d<C2D_5__IN_LINES, C2D_5__IN_COLS, C2D_5__OUT_LINES, C2D_5__OUT_COLS>(outpad_45_a[c], kernel_5[f][c], biasVal, outpad_45_b[f]);
    #else
        conv_t biasVal = bias_5_loaded[f];
        conv2d<C2D_5__IN_LINES, C2D_5__IN_COLS, C2D_5__OUT_LINES, C2D_5__OUT_COLS>(outpad_45_a[c], kernel_5_loaded[f][c], biasVal, outpad_45_b[f]);
    #endif
        ++c;
        for (; c < CHANNELS; ++c) {
        #ifndef USE_WFILES
            conv2d_multi<C2D_5__IN_LINES, C2D_5__IN_COLS, C2D_5__OUT_LINES, C2D_5__OUT_COLS>(outpad_45_a[c], outpad_45_b[f], kernel_5[f][c], outpad_45_b[f]);
        #else
            conv2d_multi<C2D_5__IN_LINES, C2D_5__IN_COLS, C2D_5__OUT_LINES, C2D_5__OUT_COLS>(outpad_45_a[c], outpad_45_b[f], kernel_5_loaded[f][c], outpad_45_b[f]);
        #endif
        }
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
    #ifndef USE_WFILES
        bnorm<BNORM_5__IN_LINES, BNORM_5__IN_COLS>(outpad_45_b[c], gamma_5[c], beta_5[c], movingmean_5[c], movingvariance_5[c], outpad_45_a[c]);
    #else
        bnorm<BNORM_5__IN_LINES, BNORM_5__IN_COLS>(outpad_45_b[c], gamma_5_loaded[c], beta_5_loaded[c], movingmean_5_loaded[c], movingvariance_5_loaded[c], outpad_45_a[c]);
    #endif
    }
    // MaxPool2D
    for (int c = 0; c < CHANNELS; ++c) {    /* 2 */
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
    for (int i = 0; i < GRU_0__IN_LINES; ++i) {
        for (int idx = 0; idx < GRU_0__IN_COLS; ++idx) {
        #ifndef USE_WFILES
            gru<GRU_0__IN_COLS, GRU_0__KERNEL_LINES, GRU_0__KERNEL_COLS, GRU_0__KERNEL_R_LINES, GRU_0__KERNEL_R_COLS, GRU_0__BIAS_SIZE>
                (idx, out_rmax0[i], kernel_gru0_f, bias_gru0_f, recurrent_kernel_gru0_f, recurrent_bias_gru0_f, &outgru_0[i][idx]);
        #else
            gru<GRU_0__IN_COLS, GRU_0__KERNEL_LINES, GRU_0__KERNEL_COLS, GRU_0__KERNEL_R_LINES, GRU_0__KERNEL_R_COLS, GRU_0__BIAS_SIZE>
                (idx, out_rmax0[i], kernel_gru0_f_loaded, bias_gru0_f_loaded, recurrent_kernel_gru0_f_loaded, recurrent_bias_gru0_f_loaded, &outgru_0[i][idx]);
        #endif
        }
        gru_syncState();
    }
    // GRU 0 (backward)
    gru_clearState();
    for (int i = GRU_0__IN_LINES-1; i >= 0; --i) {
        for (int idx = 0; idx < GRU_0__IN_COLS; ++idx) {
        #ifndef USE_WFILES
            gru<GRU_0__IN_COLS, GRU_0__KERNEL_LINES, GRU_0__KERNEL_COLS, GRU_0__KERNEL_R_LINES, GRU_0__KERNEL_R_COLS, GRU_0__BIAS_SIZE>
                (idx, out_rmax0[i], kernel_gru0_b, bias_gru0_b, recurrent_kernel_gru0_b, recurrent_bias_gru0_b, &outgru_0[i][idx+64]);
        #else
            gru<GRU_0__IN_COLS, GRU_0__KERNEL_LINES, GRU_0__KERNEL_COLS, GRU_0__KERNEL_R_LINES, GRU_0__KERNEL_R_COLS, GRU_0__BIAS_SIZE>
                (idx, out_rmax0[i], kernel_gru0_b_loaded, bias_gru0_b_loaded, recurrent_kernel_gru0_b_loaded, recurrent_bias_gru0_b_loaded, &outgru_0[i][idx + 64]);
        #endif
        }
        gru_syncState();
    }

    /* 8 */
    // GRU 1 (forward)
    gru_clearState();
    for (int i = 0; i < GRU_1__IN_LINES; ++i) {
        for (int idx = 0; idx < 64 /*GRU_1__IN_COLS/2*/; ++idx) {
        #ifndef USE_WFILES
            gru<GRU_1__IN_COLS, GRU_1__KERNEL_LINES, GRU_1__KERNEL_COLS, GRU_1__KERNEL_R_LINES, GRU_1__KERNEL_R_COLS, GRU_1__BIAS_SIZE>
                (idx, outgru_0[i], kernel_gru1_f, bias_gru1_f, recurrent_kernel_gru1_f, recurrent_bias_gru1_f, &outgru_1[i][idx]);
        #else
            gru<GRU_1__IN_COLS, GRU_1__KERNEL_LINES, GRU_1__KERNEL_COLS, GRU_1__KERNEL_R_LINES, GRU_1__KERNEL_R_COLS, GRU_1__BIAS_SIZE>
                (idx, outgru_0[i], kernel_gru1_f_loaded, bias_gru1_f_loaded, recurrent_kernel_gru1_f_loaded, recurrent_bias_gru1_f_loaded, &outgru_1[i][idx]);
        #endif
        }
        gru_syncState();
    }
    // GRU 1 (backward)
    gru_clearState();
    for (int i = GRU_1__IN_LINES-1; i >= 0; --i) {
        for (int idx = 0; idx < GRU_1__IN_COLS_BACK; ++idx) {
        #ifndef USE_WFILES
            gru<GRU_1__IN_COLS, GRU_1__KERNEL_LINES, GRU_1__KERNEL_COLS, GRU_1__KERNEL_R_LINES, GRU_1__KERNEL_R_COLS, GRU_1__BIAS_SIZE>
                (idx, outgru_0[i], kernel_gru1_b, bias_gru1_b, recurrent_kernel_gru1_b, recurrent_bias_gru1_b, &outgru_1[i][idx+64]);
        #else
            gru<GRU_1__IN_COLS, GRU_1__KERNEL_LINES, GRU_1__KERNEL_COLS, GRU_1__KERNEL_R_LINES, GRU_1__KERNEL_R_COLS, GRU_1__BIAS_SIZE>
                (idx, outgru_0[i], kernel_gru1_b_loaded, bias_gru1_b_loaded, recurrent_kernel_gru1_b_loaded, recurrent_bias_gru1_b_loaded, &outgru_1[i][idx + 64]);
        #endif
        }
        gru_syncState();
    }

    /* 9 */
    // TimeDistribution 0 (Dense)
    for (int i = 0; i < INPUT_LINES; ++i) {
    #ifndef USE_WFILES
        timedistributed_dense<TD_0__IN_LINES, TD_0__IN_COLS, TD_0__KERNEL_LINES, TD_0__KERNEL_COLS, TD_0__BIAS_SIZE, TD_0__OUT_LINES, TD_0__OUT_COLS>
            (outgru_1[i], kernel_td0, bias_td0, outtd_0[i]);
    #else
        timedistributed_dense<TD_0__IN_LINES, TD_0__IN_COLS, TD_0__KERNEL_LINES, TD_0__KERNEL_COLS, TD_0__BIAS_SIZE, TD_0__OUT_LINES, TD_0__OUT_COLS>
            (outgru_1[i], kernel_td0_loaded, bias_td0_loaded, outtd_0[i]);
    #endif
    }
    // TimeDistribution 1 (Dense)
    for (int i = 0; i < INPUT_LINES; ++i) {
    #ifndef USE_WFILES
        timedistributed_dense<TD_1__IN_LINES, TD_1__IN_COLS, TD_1__KERNEL_LINES, TD_1__KERNEL_COLS, TD_1__BIAS_SIZE, TD_1__OUT_LINES, TD_1__OUT_COLS>
            (outtd_0[i], kernel_td1, bias_td1, outputLS[i]);
    #else
        timedistributed_dense<TD_1__IN_LINES, TD_1__IN_COLS, TD_1__KERNEL_LINES, TD_1__KERNEL_COLS, TD_1__BIAS_SIZE, TD_1__OUT_LINES, TD_1__OUT_COLS>
            (outtd_0[i], kernel_td1_loaded, bias_td1_loaded, outputLS[i]);
    #endif
    }
    
    /* 10 */
    reducemax_1<RMAX_1__IN_LINES>(*outputLS, outputGS);
}



#include "z_outputexpected/dataout_0.h"
#include <time.h>
int main() {

    output_t out_local[OUTPUT_LOCAL_SCORE_LINES][OUTPUT_LOCAL_SCORE_COLS] = { 0 };
    output_t out_global[OUTPUT_GLOBAL_SCORE] = { 0 };

#ifdef USE_WFILES
    loadWeights();
#endif

    clock_t start, end;
    double cpu_time_used;
    start = clock();

    predict(input, out_local, out_global);

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        
    printf("LOCAL SCORE:\n");
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int i = 0; i < OUTPUT_LOCAL_SCORE_LINES; ++i) {
        for (int j = 0; j < OUTPUT_LOCAL_SCORE_COLS; ++j) {
            output_t out = out_local[i][j];
            output_t exp = dataoutexp_0_local[i][j];
            output_t difference = out - exp;
            special_print(out, difference, exp);
        }
    }

    printf("GLOBAL SCORE:\n");
    output_t out = out_global[0];
    output_t exp = dataoutexp_0_global[0];
    output_t difference = out - exp;
    special_print(out, difference, exp);

    printf("\nTotal time: %f seconds\n", cpu_time_used);

    return 0;
}
