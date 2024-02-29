#ifndef LOAD_WEIGHTS_H
#define LOAD_WEIGHTS_H

#include <stdio.h>

#include "types.h"
#include "size_conv3D.h"
#include "size_bgru.h"

/******************* SELECT INPUT *******************/
#define INPUT_BIRD_0 0      // expected 1 cell: 0.9706467986106873   | expected 64 cells: 0.9919541478157043
#define INPUT_BIRD_1 1      // expected 1 cell: 0.5680659413337708   | expected 64 cells: 0.9428370594978333
#define INPUT_BIRD_2 2      // expected 1 cell: 0.9324995875358582   | expected 64 cells: 0.9812303781509399
#define INPUT_BIRD_3 3      // expected 1 cell: 0.9636806845664978   | expected 64 cells: 0.9797500371932983

#define INPUT_NO_BIRD_0 4   // expected 1 cell: 0.041791435331106186 | expected 64 cells: 0.0710141733288765
#define INPUT_NO_BIRD_1 5   // expected 1 cell: 0.40294381976127625  | expected 64 cells: 0.12189839780330658
#define INPUT_NO_BIRD_2 6   // expected 1 cell: 0.061949472874403    | expected 64 cells: 0.041453707963228226
#define INPUT_NO_BIRD_3 7   // expected 1 cell: 0.12713807821273804  | expected 64 cells: 0.0648108646273613

// SELECT INPUT
#define SELECTED_INPUT INPUT_BIRD_0

#if SELECTED_INPUT == INPUT_BIRD_0
    #define SELECTED_INPUT_PATH "bird_0_50124"
#elif SELECTED_INPUT == INPUT_BIRD_1
    #define SELECTED_INPUT_PATH "bird_1_52046"
#elif SELECTED_INPUT == INPUT_BIRD_2
    #define SELECTED_INPUT_PATH "bird_2_16835"
#elif SELECTED_INPUT == INPUT_BIRD_3
    #define SELECTED_INPUT_PATH "bird_3_80705"
#elif SELECTED_INPUT == INPUT_NO_BIRD_0
    #define SELECTED_INPUT_PATH "nobird_0_50678"
#elif SELECTED_INPUT == INPUT_NO_BIRD_1
    #define SELECTED_INPUT_PATH "nobird_1_51034"
#elif SELECTED_INPUT == INPUT_NO_BIRD_2
    #define SELECTED_INPUT_PATH "nobird_2_1631"
#elif SELECTED_INPUT == INPUT_NO_BIRD_3
    #define SELECTED_INPUT_PATH "nobird_3_79266"
#else
    #error "Invalid INPUT definition"
#endif
/****************************************************/

#define SOURCE_PATH		  "D:\\BAD_FPGA\\1_vitis_hls\\mai__gC1_hcW_rnn\\source\\"

#define INPUTS_PATH       SOURCE_PATH"inputs\\"SELECTED_INPUT_PATH"\\"
#define WEIGHTS_PATH      SOURCE_PATH"weights\\"


// INPUT and OUTPUT (for validation)
#define INPUT             INPUTS_PATH"1__q_activation.bin"
#define OUTPUT_CONV       INPUTS_PATH"17__tf.math.reduce_max.bin"
#define OUTPUT_GRU0       INPUTS_PATH"18__q_bidirectional.bin"
#define OUTPUT_GRU1       INPUTS_PATH"19__q_bidirectional_1.bin"
#define OUTPUT_LS         INPUTS_PATH"21__time_distributed_1.bin"
#define OUTPUT_GS         INPUTS_PATH"22__tf.math.reduce_max_1.bin"


// timedist_0
#define TDIST_0_KERNEL WEIGHTS_PATH"time_distributed_kernel.bin"
#define TDIST_0_BIAS   WEIGHTS_PATH"time_distributed_bias.bin"

// timedist_1
#define TDIST_1_KERNEL WEIGHTS_PATH"time_distributed_1_kernel.bin"
#define TDIST_1_BIAS   WEIGHTS_PATH"time_distributed_1_bias.bin"


#define DEBUG_PRINT_LOAD


#ifdef DEBUG_PRINT_LOAD
#define PRINT_ARRAY_3D(label, outarray, dim1, dim2, dim3) \
    do { \
        quant_t(*out)[dim2][dim3] = (quant_t(*)[dim2][dim3])outarray; \
        printf("%s:\n", label); \
        for (int a = 0; a < dim1; ++a) { \
            for (int b = 0; b < dim2; ++b) { \
            	printf("  [%d][%d] > ", a, b); \
                for (int c = 0; c < dim3; ++c) { \
                    printf("%12.8f ", out[a][b][c].to_float()); \
                } \
                printf("\n"); \
            } \
			printf("  ---------\n"); \
        } \
		printf("\n"); \
    } while (0)
#define PRINT_ARRAY_4D(label, outarray, dim1, dim2, dim3, dim4) \
    do { \
        quant_t(*out)[dim2][dim3][dim4] = (quant_t(*)[dim2][dim3][dim4])outarray; \
        printf("%s:\n", label); \
        for (int a = 0; a < dim1; ++a) { \
            for (int b = 0; b < dim2; ++b) { \
                for (int c = 0; c < dim3; ++c) { \
                    printf("  [%d][%d][%d] > ", a, b, c); \
                    for (int d = 0; d < dim4; ++d) { \
                        printf("%12.8f ", out[a][b][c][d].to_float()); \
                    } \
                    printf("\n"); \
                } \
            } \
			printf("  ---------\n"); \
        } \
		printf("\n"); \
    } while (0)
#else
#define PRINT_ARRAY_3D(label, outarray, dim1, dim2, dim3) ;
#define PRINT_ARRAY_4D(label, outarray, dim1, dim2, dim3, dim4) ;
#endif


#define FILE_READ_BYTE_MULT sizeof(char)/(sizeof(weigth_t)/PACKET)

void load(const char* path, void* array, int arraysize, int typeSize) {
    FILE* file = fopen(path, "rb");

    int idx = 0;
    if (file != NULL) {
        // Read the binary data into the array
        char* buffer = (char*)array;
        fread(buffer, typeSize, arraysize, file);
        fclose(file);
        
        printf("Loaded: %s\n", path);
    }
    else {
        printf("Error loading: %s\n", path);
    }
}

void loadIO(
	imap_t* input,
	float* outputConv,
	float* outputGRU0,
	float* outputGRU1,
	float* outputLS,
	float* outputGS
) {
	load(INPUT, input, IHEIGHT*IWIDTH/PACKET_CNN, sizeof(imap_t));
	load(OUTPUT_CONV, outputConv, IHEIGHT*FILTERS, sizeof(float));
	load(OUTPUT_GRU0, outputGRU0, IHEIGHT*(GRU_FILTERS*2), sizeof(float));
	load(OUTPUT_GRU1, outputGRU1, IHEIGHT*(GRU_FILTERS*2), sizeof(float));
	load(OUTPUT_LS, outputLS, IHEIGHT, sizeof(float));
	load(OUTPUT_GS, outputGS, 1, sizeof(float));
}

void loadWeights(
	float* td0_kernel, float* td0_bias,
	float* td1_kernel, float* td1_bias
) {
    // timedist_0
    load(TDIST_0_KERNEL, td0_kernel, (FILTERS*2)*FILTERS, sizeof(float));
    load(TDIST_0_BIAS, td0_bias, FILTERS, sizeof(float));

    // timedist_1
    load(TDIST_1_KERNEL, td1_kernel, FILTERS*1, sizeof(float));
    load(TDIST_1_BIAS, td1_bias, 1, sizeof(float));
}

#endif // LOAD_WEIGHTS_H
