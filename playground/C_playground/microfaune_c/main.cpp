#include <stdio.h>
#include <string.h>

#include "input/input.h"

#include "conv2d/conv2d.h"
#include "bnorm/bnorm.h"
#include "mpool2d/maxpool2d.h"
#include "reducemax/reducemax.h"
#include "gru/gru.h"


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


#define OUT_MAX_PRINT   3
#include "conv2d/data/conv2d_0_outex.h"
#include "bnorm/data/bnorm_0_outex.h"

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
float outpad_45_trans[CHANNELS][MP2D_2__OUT_LINES][MP2D_2__OUT_COLS] = { 0 };
float outpad_45_nopad[CHANNELS][MP2D_2__RAW_OUT_LINES][MP2D_2__RAW_OUT_COLS] = { 0 };

/* 6 */
float out_rmax0[RMAX_0__OUT_LINES][RMAX_0__OUT_COLS] = { 0 };
float out_rmax0_transposed[RMAX_0__OUT_COLS][RMAX_0__OUT_LINES] = { 0.0 };

/* 7 */
float outgru_0[GRU_0__OUT_LINES][GRU_0__OUT_COLS] = { 0.0 };

/* 8 */
float outgru_1[GRU_1__OUT_LINES][GRU_1__OUT_COLS] = { 0.0 };



void predict(const input_t input[INPUT_LINES][INPUT_COLS]/*, output_t output*/) {
    input_preconv2d(input, inputpad);

    /*************************************/
    /**************** CNN ****************/
    /*************************************/
    /* 0 */
    // Conv2D
    for (int c = 0; c < CHANNELS; ++c) {
        conv2d<C2D_0__IN_LINES, C2D_0__IN_COLS, C2D_0__OUT_LINES, C2D_0__OUT_COLS>(inputpad, kernel_0[c], bias_0[c], outpad_01_a[c]);
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_0__IN_LINES, BNORM_0__IN_COLS>(outpad_01_a[c], gamma_0[c], beta_0[c], movingmean_0[c], movingvariance_0[c], outpad_01_b[c]);
    }

    /* 1 */
    // Conv2D
    for (int c = 0; c < CHANNELS; ++c) {
        conv_t biasVal = bias_1[c];
        int f = 0;
        conv2d<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_01_b[f], kernel_1[f][c], biasVal, outpad_01_a[c]);
        ++f;
        for (; f < FILTERS; ++f) {
            conv2d_multi<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_01_b[f], outpad_01_a[c], kernel_1[f][c], 0, outpad_01_a[c]);
        }
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_1__IN_LINES, BNORM_1__IN_COLS>(outpad_01_a[c], gamma_1[c], beta_1[c], movingmean_1[c], movingvariance_1[c], outpad_01_b[c]);
    }
    // MaxPool2D
    for (int c = 0; c < CHANNELS; ++c) {    /* 0 */
        maxpool2d<MP2D_0__IN_LINES, MP2D_0__IN_COLS, MP2D_0__OUT_LINES, MP2D_0__OUT_COLS>(outpad_01_b[c], outpad_23_a[c]);
    }

    /* 2 */
    // Conv2D
    for (int c = 0; c < CHANNELS; ++c) {
        conv_t biasVal = bias_2[c];
        int f = 0;
        conv2d<C2D_2__IN_LINES, C2D_2__IN_COLS, C2D_2__OUT_LINES, C2D_2__OUT_COLS>(outpad_23_a[f], kernel_2[f][c], biasVal, outpad_23_b[c]);
        ++f;
        for (; f < FILTERS; ++f) {
            conv2d_multi<C2D_2__IN_LINES, C2D_2__IN_COLS, C2D_2__OUT_LINES, C2D_2__OUT_COLS>(outpad_23_a[f], outpad_23_b[c], kernel_2[f][c], 0, outpad_23_b[c]);
        }
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_2__IN_LINES, BNORM_2__IN_COLS>(outpad_23_b[c], gamma_2[c], beta_2[c], movingmean_2[c], movingvariance_2[c], outpad_23_a[c]);
    }

    /* 3 */
    // Conv2D
    for (int c = 0; c < CHANNELS; ++c) {
        conv_t biasVal = bias_3[c];
        int f = 0;
        conv2d<C2D_3__IN_LINES, C2D_3__IN_COLS, C2D_3__OUT_LINES, C2D_3__OUT_COLS>(outpad_23_a[f], kernel_3[f][c], biasVal, outpad_23_b[c]);
        ++f;
        for (; f < FILTERS; ++f) {
            conv2d_multi<C2D_3__IN_LINES, C2D_3__IN_COLS, C2D_3__OUT_LINES, C2D_3__OUT_COLS>(outpad_23_a[f], outpad_23_b[c], kernel_3[f][c], 0, outpad_23_b[c]);
        }
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_3__IN_LINES, BNORM_3__IN_COLS>(outpad_23_b[c], gamma_3[c], beta_3[c], movingmean_3[c], movingvariance_3[c], outpad_23_a[c]);
    }
    // MaxPool2D
    for (int c = 0; c < CHANNELS; ++c) {    /* 1 */
        maxpool2d<MP2D_1__IN_LINES, MP2D_1__IN_COLS, MP2D_1__OUT_LINES, MP2D_1__OUT_COLS>(outpad_23_a[c], outpad_45_a[c]);
    }

    /* 4 */
    // Conv2D
    for (int c = 0; c < CHANNELS; ++c) {
        conv_t biasVal = bias_4[c];
        int f = 0;
        conv2d<C2D_4__IN_LINES, C2D_4__IN_COLS, C2D_4__OUT_LINES, C2D_4__OUT_COLS>(outpad_45_a[f], kernel_4[f][c], biasVal, outpad_45_b[c]);
        ++f;
        for (; f < FILTERS; ++f) {
            conv2d_multi<C2D_4__IN_LINES, C2D_4__IN_COLS, C2D_4__OUT_LINES, C2D_4__OUT_COLS>(outpad_45_a[f], outpad_45_b[c], kernel_4[f][c], 0, outpad_45_b[c]);
        }
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_4__IN_LINES, BNORM_4__IN_COLS>(outpad_45_b[c], gamma_4[c], beta_4[c], movingmean_4[c], movingvariance_4[c], outpad_45_a[c]);
    }

    /* 5 */
    // Conv2D
    for (int c = 0; c < CHANNELS; ++c) {
        conv_t biasVal = bias_5[c];
        int f = 0;
        conv2d<C2D_5__IN_LINES, C2D_5__IN_COLS, C2D_5__OUT_LINES, C2D_5__OUT_COLS>(outpad_45_a[f], kernel_5[f][c], biasVal, outpad_45_b[c]);
        ++f;
        for (; f < FILTERS; ++f) {
            conv2d_multi<C2D_5__IN_LINES, C2D_5__IN_COLS, C2D_5__OUT_LINES, C2D_5__OUT_COLS>(outpad_45_a[f], outpad_45_b[c], kernel_5[f][c], 0, outpad_45_b[c]);
        }
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_5__IN_LINES, BNORM_5__IN_COLS>(outpad_45_b[c], gamma_5[c], beta_5[c], movingmean_5[c], movingvariance_5[c], outpad_45_a[c]);
    }
    // MaxPool2D
    for (int c = 0; c < CHANNELS; ++c) {    /* 1 */
        maxpool2d<MP2D_2__IN_LINES, MP2D_2__IN_COLS, MP2D_2__OUT_LINES, MP2D_2__OUT_COLS>(outpad_45_a[c], outpad_45_trans[c]);
    }

    /* REMOVE PADDING: Transition from CNN to RNN */
    for (int c = 0; c < CHANNELS; ++c) {
        for (int row = PADDING_OFFSET; row < MP2D_2__OUT_LINES; ++row) {
            for (int col = PADDING_OFFSET; col < MP2D_2__OUT_COLS; ++col) {
                outpad_45_nopad[c][row-1][col-1] = outpad_45_trans[c][row][col];
            }
        }
    }

    /* 6 */
    // ReduceMax
    reducemax_0<RMAX_0__IN_LINES, RMAX_0__IN_COLS, RMAX_0__OUT_LINES, RMAX_0__OUT_COLS>(outpad_45_nopad, out_rmax0);

    /* TRANSPOSE: From channel first to channel last */
    for (int row = 0; row < RMAX_0__OUT_LINES; ++row) {
        for (int col = 0; col < RMAX_0__OUT_COLS; ++col) {
            out_rmax0_transposed[col][row] = out_rmax0[row][col];
        }
    }

    /*************************************/
    /**************** RNN ****************/
    /*************************************/
    /* 7 */
    // GRU 0 (forward)
    gru_clearState();
    for (int i = 0; i < GRU_0__IN_LINES; ++i) {
        for (int idx = 0; idx < GRU_0__IN_COLS; ++idx) {
            gru<GRU_0__IN_COLS, GRU_0__KERNEL_LINES, GRU_0__KERNEL_COLS, GRU_0__KERNEL_R_LINES, GRU_0__KERNEL_R_COLS, GRU_0__BIAS_SIZE>
                (idx, out_rmax0_transposed[i], kernel_gru0_f, bias_gru0_f, recurrent_kernel_gru0_f, recurrent_bias_gru0_f, &outgru_0[i][idx]);
        }
        gru_syncState();
    }
    // GRU 0 (backward)
    gru_clearState();
    for (int i = GRU_0__IN_LINES-1; i >= 0; --i) {
        for (int idx = 0; idx < GRU_0__IN_COLS; ++idx) {
            gru<GRU_0__IN_COLS, GRU_0__KERNEL_LINES, GRU_0__KERNEL_COLS, GRU_0__KERNEL_R_LINES, GRU_0__KERNEL_R_COLS, GRU_0__BIAS_SIZE>
                (idx, out_rmax0_transposed[i], kernel_gru0_b, bias_gru0_b, recurrent_kernel_gru0_b, recurrent_bias_gru0_b, &outgru_0[i][idx+64]);
        }
        gru_syncState();
    }

    /* 8 */
    // GRU 1 (forward)
    gru_clearState();
    for (int i = 0; i < GRU_1__IN_LINES; ++i) {
        for (int idx = 0; idx < GRU_1__IN_COLS; ++idx) {
            gru<GRU_1__IN_COLS, GRU_1__KERNEL_LINES, GRU_1__KERNEL_COLS, GRU_1__KERNEL_R_LINES, GRU_1__KERNEL_R_COLS, GRU_1__BIAS_SIZE>
                (idx, outgru_0[i], kernel_gru1_f, bias_gru1_f, recurrent_kernel_gru1_f, recurrent_bias_gru1_f, &outgru_1[i][idx]);
        }
        gru_syncState();
    }
    // GRU 1 (backward)
    gru_clearState();
    for (int i = GRU_1__IN_LINES-1; i >= 0; --i) {
        for (int idx = 0; idx < GRU_1__IN_COLS; ++idx) {
            gru<GRU_1__IN_COLS, GRU_1__KERNEL_LINES, GRU_1__KERNEL_COLS, GRU_1__KERNEL_R_LINES, GRU_1__KERNEL_R_COLS, GRU_1__BIAS_SIZE>
                (idx, outgru_0[i], kernel_gru1_b, bias_gru1_b, recurrent_kernel_gru1_b, recurrent_bias_gru1_b, &outgru_1[i][idx+64]);
        }
        gru_syncState();
    }


    



    printf("      OUTPUT\n");
    for (int outter = 63; outter < 64; ++outter) {
        printf("[%3d] ##############################################\n", outter);
        for (int i = 0; i < 431; ++i) {
            bnorm_t out = out_rmax0[outter][i];
            special_print(out, 0, 0);
        }
        printf("\n");
    }


    /*
    // Print the output values
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int i = 1; i < (40+1); ++i) {
        bnorm_t out = outpad_c[0][1][i];
        bnorm_t exp = expt[i - 1];
        bnorm_t difference = out - exp;
        //printf(" %15.12f | %7.4f | %15.12f\n", out, difference, exp);
        special_print(out, difference, exp);
    }
    */

    /*
    // Print the output values
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int outter = 1; outter < OUT_MAX_PRINT; ++outter) {
        printf("[%3d] ##############################################\n", outter);
        for (int i = 1; i < INPUT_COLS; ++i) {
            bnorm_t out = outputpad[0][outter][i];
            bnorm_t exp = bnorm_output_expected_channel0[outter - 1][i - 1];
            bnorm_t difference = out - exp;
            //printf(" %15.12f | %7.4f | %15.12f\n", out, difference, exp);
            special_print(out, difference, exp);
        }
        printf("\n");
    }
    */
}








int main() {

    predict(input);
        
    return 0;
}
