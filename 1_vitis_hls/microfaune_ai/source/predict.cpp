#include "out_expect.h"


#include <stdio.h>
#include "types.h"

#include "input/input.h"

#include "conv2d/conv2d.h"

/* 0 */
#include "conv2d/data/conv2d_0.h"

/* 1 */
#include "conv2d/data/conv2d_1.h"


#include "load_weights.h"


conv_t outarray_a[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS] = { 0 };
//conv_t outarray_b[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS] = { 0 };

input_t inputpad[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS] = { 0 };

/* 10 */
// outputLS
// outputGS

//#define DEBUG_PRINT_OUTPUT
//#define DEBUG_GRU0_TO_FILE
//#define DEBUG_CLEAR_OUTPUT_BEFORE_GRU0

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
    /*
    output_t outputLS[OUTPUT_LOCAL_SCORE_LINES][OUTPUT_LOCAL_SCORE_COLS],
    output_t outputGS[OUTPUT_GLOBAL_SCORE]
    */
    conv_t outarray_b[CHANNELS][C2D_0__IN_LINES][C2D_0__IN_COLS]
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
    
}
