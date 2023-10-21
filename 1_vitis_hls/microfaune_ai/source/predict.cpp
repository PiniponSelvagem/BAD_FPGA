#include "out_expect.h"




#include <stdio.h>

#include "types.h"

#include "input/input.h"
#include "output/output.h"

#include "conv2d/conv2d.h"
#include "mpool2d/maxpool2d.h"
#include "reducemax/reducemax.h"
#include "gru/gru.h"
#include "timedist/timedist.h"

/* 0 */
#include "conv2d/data/conv2d_0.h"

/* 1 */
#include "conv2d/data/conv2d_1.h"
#include "mpool2d/data/maxpool2d_0.h"

/* 2 */
#include "conv2d/data/conv2d_2.h"

/* 3 */
#include "conv2d/data/conv2d_3.h"
#include "mpool2d/data/maxpool2d_1.h"

/* 4 */
#include "conv2d/data/conv2d_4.h"

/* 5 */
#include "conv2d/data/conv2d_5.h"
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

input_t inputpad[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS] = { 0 };

/* 10 */
// outputLS
// outputGS

//#define DEBUG_PRINT_OUTPUT
#define D_C 0
#ifdef DEBUG_PRINT_OUTPUT
#ifdef __VITIS_HLS__
#define DEBUG_PRINT(label, outarray, dim1, dim2, dim3) \
    do { \
        conv_t(*out)[dim2][dim3] = (conv_t(*)[dim2][dim3])outarray; \
        printf("%s:\n", label); \
        for (int b = 0; b < 4; ++b) { \
            for (int c = 0; c < 4; ++c) { \
                printf("%12.8f ", out[D_C][b][c].to_float()); \
            } \
            printf("\n"); \
        } \
    } while (0)
#else
#define DEBUG_PRINT(label, outarray, dim1, dim2, dim3) \
    do { \
        conv_t(*out)[dim2][dim3] = (conv_t(*)[dim2][dim3])outarray; \
        printf("%s:\n", label); \
        for (int b = 0; b < 4; ++b) { \
            for (int c = 0; c < 4; ++c) { \
                printf("%12.8f ", out[D_C][b][c]); \
            } \
            printf("\n"); \
        } \
    } while (0)
#endif
#else
#define DEBUG_PRINT(label, outarray, dim1, dim2, dim3) ;
#endif


void predict(
    const input_t input[INPUT_LINES][INPUT_COLS],
    output_t outputLS[OUTPUT_LOCAL_SCORE_LINES][OUTPUT_LOCAL_SCORE_COLS],
    output_t outputGS[OUTPUT_GLOBAL_SCORE]
) {
#pragma HLS INTERFACE s_axilite port=input bundle=BUS1
#pragma HLS INTERFACE s_axilite port=outputLS bundle=BUS1
#pragma HLS INTERFACE s_axilite port=outputGS bundle=BUS1
#pragma HLS INTERFACE s_axilite port=return bundle=BUS1

    input_preconv2d(input, inputpad);
    /*************************************/
    /**************** CNN ****************/
    /*************************************/
    /* 0 */
    conv2d( // CONV2D_0
        C2D_0_FILTERS,
        C2D_0__IN_COLS,
        (conv_t*)inputpad,
        (conv_t*)kernel_0,
        (conv_t*)bias_0,
        (conv_t*)outarray_a
    );
    DEBUG_PRINT("CONV2D_0", outarray_a, 64, 433, 42);

    /* 1 */
    conv2d( // CONV2D_1
        FILTERS,
        C2D_1__IN_COLS,
        (conv_t*)outarray_a,
        (conv_t*)kernel_1,
        (conv_t*)bias_1,
        (conv_t*)outarray_b
    );
    DEBUG_PRINT("CONV2D_1", outarray_b, 64, 433, 42);
    maxpool2d( // MPOOL2D_0
        MP2D_0__IN_COLS,
        MP2D_0__OUT_COLS,
        PADDING_OFFSET,
        (mpool_t*)outarray_b
    );
    DEBUG_PRINT("MPOOL2D_0", outarray_b, 64, 433, 22);

    /* 2 */
    conv2d( // CONV2D_2
        FILTERS,
        C2D_2__IN_COLS,
        (conv_t*)outarray_b,
        (conv_t*)kernel_2,
        (conv_t*)bias_2,
        (conv_t*)outarray_a
    );
    DEBUG_PRINT("CONV2D_2", outarray_a, 64, 433, 22);

    /* 3 */
    conv2d( // CONV2D_3
        FILTERS,
        C2D_3__IN_COLS,
        (conv_t*)outarray_a,
        (conv_t*)kernel_3,
        (conv_t*)bias_3,
        (conv_t*)outarray_b
    );
    DEBUG_PRINT("CONV2D_3", outarray_b, 64, 433, 22);
    maxpool2d( // MPOOL2D_1
        MP2D_1__IN_COLS,
        MP2D_1__OUT_COLS,
        PADDING_OFFSET,
        (mpool_t*)outarray_b
    );
    DEBUG_PRINT("MPOOL2D_1", outarray_b, 64, 433, 12);
    
    /* 4 */
    conv2d( // CONV2D_4
        FILTERS,
        C2D_4__IN_COLS,
        (conv_t*)outarray_b,
        (conv_t*)kernel_4,
        (conv_t*)bias_4,
        (conv_t*)outarray_a
    );
    DEBUG_PRINT("CONV2D_4", outarray_a, 64, 433, 12);

    /* 5 */
    conv2d( // CONV2D_5
        FILTERS,
        C2D_5__IN_COLS,
        (conv_t*)outarray_a,
        (conv_t*)kernel_5,
        (conv_t*)bias_5,
        (conv_t*)outarray_b
    );
    DEBUG_PRINT("CONV2D_5", outarray_b, 64, 433, 12);
    maxpool2d( // MPOOL2D_2
        MP2D_2__IN_COLS,
        MP2D_2__OUT_COLS,
        0,
        (mpool_t*)outarray_b
    );
    DEBUG_PRINT("MPOOL2D_2", outarray_b, 64, 433, 5);

    /* 6 */
    reducemax_0_saveTranspose( // RMAX_0 + Save transposed
        (reducemax_t*)outarray_b,
        (reducemax_t*)outarray_a
    );
    DEBUG_PRINT("RMAX_0", outarray_a, 1, 433, 64);


#ifdef DEBUG_CNN
    /* CHECK output expected for data_bird_0 */
    float* pout_a = (float*)outarray_a;
    float* pexp_a = (float*)outexp_before_rnn;
    int errorCount = 0;
    for (int i = 0; i<(431*64); ++i) {
        if (*pout_a != *pexp_a) {
            printf("out | diff | exp  : %7.4f | %7.4f | %7.4f\n", *pout_a, *pout_a - *pexp_a, *pexp_a);
            ++errorCount;
        }
        ++pout_a;
        ++pexp_a;
    }
    float errorPercent = ((float)errorCount / (431 * 64)) * 100;
    printf("ERROR COUNT = %d of %d\n", errorCount, (431*64));
    printf("ERROR PRCNT = %f %\n", errorPercent);

    /* CLEAR outarray_b */
    /* TEMPORARY ONLY FOR DEBUG, can be safely removed */
    for (int a = 0; a < 64; ++a) {
        for (int b = 0; b < 433; ++b) {
            for (int c = 0; c < 42; ++c) {
                outarray_b[a][b][c] = 88;
            }
        }
    }
#endif


    /*************************************/
    /**************** RNN ****************/
    /*************************************/
    /* 7 */
    gru( // GRU_0_F
        GRU_FORWARD,
        GRU_0__IN_COLS, (RNN_COLS_GRU/2),
        GRU_0__KERNEL_COLS,
        (gru_t*)outarray_a,
        (gru_t*)kernel_gru0_f,           (gru_t*)bias_gru0_f,
        (gru_t*)recurrent_kernel_gru0_f,
        (gru_t*)outarray_b
    );
    gru( // GRU_0_B
        GRU_BACKWARD,
        GRU_0__IN_COLS, (RNN_COLS_GRU/2),
        GRU_0__KERNEL_COLS,
        (gru_t*)outarray_a,
        (gru_t*)kernel_gru0_b,           (gru_t*)bias_gru0_b,
        (gru_t*)recurrent_kernel_gru0_b,
        (gru_t*)outarray_b
    );

#ifdef DEBUG_GRU0
    /* CHECK output expected for data_bird_0 */
    float* pout_a = (float*)outarray_b;
    float* pexp_a = (float*)outexp_after_gru0;
    int errorCount_different = 0;
    int errorCount_offbyAbove0125 = 0;
    for (int i = 0; i < (431 * 128); ++i) {
        if (*pout_a != *pexp_a) {
            printf("out | diff | exp  : %7.4f | %7.4f | %7.4f\n", *pout_a, *pout_a - *pexp_a, *pexp_a);
            ++errorCount_different;
            if (*pout_a - *pexp_a > 0.125 || *pout_a - *pexp_a < -0.125)
                ++errorCount_offbyAbove0125;
        }
        ++pout_a;
        ++pexp_a;
    }
    float errorPercent_different = ((float)errorCount_different / (431 * 128)) * 100;
    printf("ERROR COUNT (different) = %d of %d\n", errorCount_different, (431 * 128));
    printf("ERROR PRCNT (different) = %f %\n", errorPercent_different);

    float errorPercent_offbyAbove0125 = ((float)errorCount_offbyAbove0125 / (431 * 128)) * 100;
    printf("ERROR COUNT (offbyAbove0125) = %d of %d\n", errorCount_offbyAbove0125, (431 * 128));
    printf("ERROR PRCNT (offbyAbove0125) = %f %\n", errorPercent_offbyAbove0125);

    /* CLEAR outarray_b */
    /* TEMPORARY ONLY FOR DEBUG, can be safely removed */
    for (int a = 0; a < 64; ++a) {
        for (int b = 0; b < 433; ++b) {
            for (int c = 0; c < 42; ++c) {
                outarray_a[a][b][c] = 88;
            }
        }
    }
#endif


    /*
    ///////////////////////////////////////////////////////////////////////////////////
    // Open the log file in write mode
    FILE* logFile = fopen("OUTPUT_C.log", "w");

    if (logFile == NULL) {
        printf("Error opening file");
    }

    gru_t* ptr = (gru_t*)outarray_b;

    // Loop through the array using pointer arithmetic and print each value to the log file
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 431; j++) {
            fprintf(logFile, "%f ", *(ptr + i * 431 + j));
        }
        fprintf(logFile, "\n"); // Newline after each row
    }

    // Close the log file
    fclose(logFile);
    /////////////////////////////////////////////////////////////////////////////////
    */

    /* 8 */
    gru( // GRU_1_F
        GRU_FORWARD,
        GRU_1__IN_COLS, RNN_COLS_GRU,
        GRU_1__KERNEL_COLS,
        (gru_t*)outarray_b,
        (gru_t*)kernel_gru1_f,           (gru_t*)bias_gru1_f,
        (gru_t*)recurrent_kernel_gru1_f,
        (gru_t*)outarray_a
    );
    gru( // GRU_1_B
        GRU_BACKWARD,
        GRU_1__IN_COLS_BACK, RNN_COLS_GRU,
        GRU_1__KERNEL_COLS,
        (gru_t*)outarray_b,
        (gru_t*)kernel_gru1_b,           (gru_t*)bias_gru1_b,
        (gru_t*)recurrent_kernel_gru1_b,
        (gru_t*)outarray_a
    );

    /* 9 */
    timedistributed_dense( // TDIST_0 + Dense
        TD_0__IN_COLS,
        TD_0__KERNEL_LINES, TD_0__KERNEL_COLS,
        TD_0__OUT_COLS,
        (timedist_t*)outarray_a,
        (timedist_t*)kernel_td0,
        (timedist_t*)bias_td0,
        (timedist_t*)outarray_b
    );
    timedistributed_dense( // TDIST_1 + Dense
        TD_1__IN_COLS,
        TD_1__KERNEL_LINES, TD_1__KERNEL_COLS,
        TD_1__OUT_COLS,
        (timedist_t*)outarray_b,
        (timedist_t*)kernel_td1,
        (timedist_t*)bias_td1,
        (timedist_t*)outputLS   // LOCAL OUTPUT
    );

    /* 10 */
    reducemax_1( // RMAX_1
        (output_t*)outputLS,    // LOCAL OUTPUT -> 431
        (output_t*)outputGS     // GLOBAL OUTPUT -> 1
    );
}
