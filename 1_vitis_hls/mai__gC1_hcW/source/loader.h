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

#define SOURCE_PATH		  "D:\\BAD_FPGA\\1_vitis_hls\\microfaune_ai__gruCells1__hardcodedWeights\\source\\"

#define INPUTS_PATH       SOURCE_PATH"inputs\\"SELECTED_INPUT_PATH"\\"
#define WEIGHTS_PATH      SOURCE_PATH"weights\\"


// INPUT and OUTPUT (for validation)
#define INPUT             INPUTS_PATH"1__q_activation.bin"
#define OUTPUT_CONV       INPUTS_PATH"17__tf.math.reduce_max.bin"
#define OUTPUT_GRU0       INPUTS_PATH"18__q_bidirectional.bin"
#define OUTPUT_GRU1       INPUTS_PATH"19__q_bidirectional_1.bin"
#define OUTPUT_LS         INPUTS_PATH"21__time_distributed_1.bin"
#define OUTPUT_GS         INPUTS_PATH"22__tf.math.reduce_max_1.bin"


// gru_0_forward
#define GRU_0_FORWARD_KERNEL                    WEIGHTS_PATH"q_bidirectional_gru_forward_kernel.bin"
#define GRU_0_FORWARD_RECURRENT_KERNEL          WEIGHTS_PATH"q_bidirectional_gru_forward_recurrent_kernel.bin"
#define GRU_0_FORWARD_BIAS                      WEIGHTS_PATH"q_bidirectional_gru_forward_bias.bin"
#define GRU_0_FORWARD_RECURRENT_BIAS            WEIGHTS_PATH"q_bidirectional_gru_forward_bias_recurrent.bin"
// gru_0_backward
#define GRU_0_BACKWARD_KERNEL                   WEIGHTS_PATH"q_bidirectional_gru_backward_kernel.bin"
#define GRU_0_BACKWARD_RECURRENT_KERNEL         WEIGHTS_PATH"q_bidirectional_gru_backward_recurrent_kernel.bin"
#define GRU_0_BACKWARD_BIAS                     WEIGHTS_PATH"q_bidirectional_gru_backward_bias.bin"
#define GRU_0_BACKWARD_RECURRENT_BIAS   		WEIGHTS_PATH"q_bidirectional_gru_backward_bias_recurrent.bin"

// gru_1_forward
#define GRU_1_FORWARD_KERNEL                    WEIGHTS_PATH"q_bidirectional_1_gru_forward_kernel.bin"
#define GRU_1_FORWARD_RECURRENT_KERNEL          WEIGHTS_PATH"q_bidirectional_1_gru_forward_recurrent_kernel.bin"
#define GRU_1_FORWARD_BIAS                      WEIGHTS_PATH"q_bidirectional_1_gru_forward_bias.bin"
#define GRU_1_FORWARD_RECURRENT_BIAS            WEIGHTS_PATH"q_bidirectional_1_gru_forward_bias_recurrent.bin"
// gru_1_backward
#define GRU_1_BACKWARD_KERNEL                   WEIGHTS_PATH"q_bidirectional_1_gru_backward_kernel.bin"
#define GRU_1_BACKWARD_RECURRENT_KERNEL         WEIGHTS_PATH"q_bidirectional_1_gru_backward_recurrent_kernel.bin"
#define GRU_1_BACKWARD_BIAS                     WEIGHTS_PATH"q_bidirectional_1_gru_backward_bias.bin"
#define GRU_1_BACKWARD_RECURRENT_BIAS   		WEIGHTS_PATH"q_bidirectional_1_gru_backward_bias_recurrent.bin"

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
	gru_weigth_t* gru0f_kernel, gru_weigth_t* gru0f_rkernel, gru_weigth_t* gru0f_bias, gru_weigth_t* gru0f_rbias,
	gru_weigth_t* gru0b_kernel, gru_weigth_t* gru0b_rkernel, gru_weigth_t* gru0b_bias, gru_weigth_t* gru0b_rbias,
	gru_weigth_t* gru1f_kernel, gru_weigth_t* gru1f_rkernel, gru_weigth_t* gru1f_bias, gru_weigth_t* gru1f_rbias,
	gru_weigth_t* gru1b_kernel, gru_weigth_t* gru1b_rkernel, gru_weigth_t* gru1b_bias, gru_weigth_t* gru1b_rbias,
	float* td0_kernel, float* td0_bias,
	float* td1_kernel, float* td1_bias
) {
    // gru_0_forward
    load(GRU_0_FORWARD_KERNEL, gru0f_kernel, GRU_FILTERS*CHANNELS*GRU_SPLIT_SIZE, G_WG_W_BIT_WIDTH);
    load(GRU_0_FORWARD_RECURRENT_KERNEL, gru0f_rkernel, GRU_FILTERS*GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
    load(GRU_0_FORWARD_BIAS, gru0f_bias, GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
    load(GRU_0_FORWARD_RECURRENT_BIAS, gru0f_rbias, GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
    // gru_0_backward
    load(GRU_0_BACKWARD_KERNEL, gru0b_kernel, GRU_FILTERS*CHANNELS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
    load(GRU_0_BACKWARD_RECURRENT_KERNEL, gru0b_rkernel, GRU_FILTERS*GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
    load(GRU_0_BACKWARD_BIAS, gru0b_bias, GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
    load(GRU_0_BACKWARD_RECURRENT_BIAS, gru0b_rbias, GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));

	// gru_1_forward
	load(GRU_1_FORWARD_KERNEL, gru1f_kernel, (GRU_FILTERS*2)*GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
	load(GRU_1_FORWARD_RECURRENT_KERNEL, gru1f_rkernel, GRU_FILTERS*GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
	load(GRU_1_FORWARD_BIAS, gru1f_bias, GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
	load(GRU_1_FORWARD_RECURRENT_BIAS, gru1f_rbias, GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
	// gru_1_backward
	load(GRU_1_BACKWARD_KERNEL, gru1b_kernel, (GRU_FILTERS*2)*GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
	load(GRU_1_BACKWARD_RECURRENT_KERNEL, gru1b_rkernel, GRU_FILTERS*GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
	load(GRU_1_BACKWARD_BIAS, gru1b_bias, GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));
	load(GRU_1_BACKWARD_RECURRENT_BIAS, gru1b_rbias, GRU_FILTERS*GRU_SPLIT_SIZE, sizeof(G_WG_W_BIT_WIDTH));

    // timedist_0
    load(TDIST_0_KERNEL, td0_kernel, (FILTERS*2)*FILTERS, sizeof(float));
    load(TDIST_0_BIAS, td0_bias, FILTERS, sizeof(float));

    // timedist_1
    load(TDIST_1_KERNEL, td1_kernel, FILTERS*1, sizeof(float));
    load(TDIST_1_BIAS, td1_bias, 1, sizeof(float));
}

#endif // LOAD_WEIGHTS_H
