#include <stdio.h>
#include <string.h>

#include "input/input.h"

#include "conv2d/conv2d.h"
#include "bnorm/bnorm.h"
#include "mpool2d/maxpool2d.h"

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

/* 2 */
float outpad_2_a[CHANNELS][MP2D_0__OUT_LINES][MP2D_0__OUT_COLS] = { 0 };
float outpad_2_b[CHANNELS][MP2D_0__OUT_LINES][MP2D_0__OUT_COLS] = { 0 };

/* 3 */
float outpad_3_a[CHANNELS][MP2D_1__OUT_LINES][MP2D_1__OUT_COLS] = { 0 };
float outpad_3_b[CHANNELS][MP2D_1__OUT_LINES][MP2D_1__OUT_COLS] = { 0 };



void predict(const input_t input[INPUT_LINES][INPUT_COLS]/*, output_t output*/) {
    input_preconv2d(input, inputpad);

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
        maxpool2d<MP2D_0__IN_LINES, MP2D_0__IN_COLS, MP2D_0__OUT_LINES, MP2D_0__OUT_COLS>(outpad_01_b[c], outpad_2_a[c]);
    }

    /* 2 */
    // Conv2D
    for (int c = 0; c < CHANNELS; ++c) {
        conv_t biasVal = bias_2[c];
        int f = 0;
        conv2d<C2D_2__IN_LINES, C2D_2__IN_COLS, C2D_2__OUT_LINES, C2D_2__OUT_COLS>(outpad_2_a[f], kernel_2[f][c], biasVal, outpad_2_b[c]);
        ++f;
        for (; f < FILTERS; ++f) {
            conv2d_multi<C2D_2__IN_LINES, C2D_2__IN_COLS, C2D_2__OUT_LINES, C2D_2__OUT_COLS>(outpad_2_a[f], outpad_2_b[c], kernel_2[f][c], 0, outpad_2_b[c]);
        }
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_2__IN_LINES, BNORM_2__IN_COLS>(outpad_2_b[c], gamma_2[c], beta_2[c], movingmean_2[c], movingvariance_2[c], outpad_2_a[c]);
    }

    /* 3 */
    // Conv2D
    for (int c = 0; c < CHANNELS; ++c) {
        conv_t biasVal = bias_3[c];
        int f = 0;
        conv2d<C2D_3__IN_LINES, C2D_3__IN_COLS, C2D_3__OUT_LINES, C2D_3__OUT_COLS>(outpad_2_a[f], kernel_3[f][c], biasVal, outpad_2_b[c]);
        ++f;
        for (; f < FILTERS; ++f) {
            conv2d_multi<C2D_3__IN_LINES, C2D_3__IN_COLS, C2D_3__OUT_LINES, C2D_3__OUT_COLS>(outpad_2_a[f], outpad_2_b[c], kernel_3[f][c], 0, outpad_2_b[c]);
        }
    }
    // BatchNormalization
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_3__IN_LINES, BNORM_3__IN_COLS>(outpad_2_b[c], gamma_3[c], beta_3[c], movingmean_3[c], movingvariance_3[c], outpad_2_a[c]);
    }
    // MaxPool2D
    for (int c = 0; c < CHANNELS; ++c) {    /* 1 */
        maxpool2d<MP2D_1__IN_LINES, MP2D_1__IN_COLS, MP2D_1__OUT_LINES, MP2D_1__OUT_COLS>(outpad_2_a[c], outpad_3_a[c]);
    }




    
    printf("      OUTPUT\n");
    for (int outter = 1; outter < 8; ++outter) {
        printf("[%3d] ##############################################\n", outter);
        for (int i = 1; i < 11; ++i) {
            bnorm_t out = outpad_3_a[0][outter][i];
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

const mpool_t test_in[2][4+2][8+2] = {
        {
            {0, 0,0,0,0,0,0,0,0, 0},

            {0, 1,2,3,4,5,6,7,8, 0},
            {0, 0,1,0,1,0,1,0,1, 0},
            {0, 2,0,2,0,2,0,2,0, 0},
            {0, 1,8,2,7,3,6,4,5, 0},

            {0, 0,0,0,0,0,0,0,0, 0},
        },
        {
            {0, 0,0,0,0,0,0,0,0, 0},

            {0, 8,2,7,3,6,4,5,3, 0},
            {0, 2,3,4,5,6,7,8,9, 0},
            {0, 0,8,0,8,0,8,0,8, 0},
            {0, 4,0,4,0,4,0,4,0, 0},

            {0, 0,0,0,0,0,0,0,0, 0},
        },
};

mpool_t test_out[2][4+2][4+2] = { 0 };





int main() {

    predict(input);
    
    return 0;
}
