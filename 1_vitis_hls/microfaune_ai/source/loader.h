#ifndef LOAD_WEIGHTS_H
#define LOAD_WEIGHTS_H

#include <stdio.h>

#include "types.h"
#include "size_conv3D.h"
#include "size_bgru.h"

/******************* SELECT INPUT *******************/
#define INPUT_BIRD_0 0      // expected: 0.9919541478157043
#define INPUT_BIRD_1 1      // expected: 0.9428370594978333
#define INPUT_BIRD_2 2      // expected: 0.9812303781509399
#define INPUT_BIRD_3 3      // expected: 0.9797500371932983

#define INPUT_NO_BIRD_0 4   // expected: 0.0710141733288765
#define INPUT_NO_BIRD_1 5   // expected: 0.12189839780330658
#define INPUT_NO_BIRD_2 6   // expected: 0.041453707963228226
#define INPUT_NO_BIRD_3 7   // expected: 0.0648108646273613

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

#define SOURCE_PATH		  "E:\\Rodrigo\\ISEL\\2_Mestrado\\2-ANO_1-sem\\TFM\\BAD_FPGA\\1_vitis_hls\\microfaune_ai\\source\\"

#define INPUTS_PATH       SOURCE_PATH"inputs\\"SELECTED_INPUT_PATH"\\"
#define WEIGHTS_PATH      SOURCE_PATH"weights\\"


// INPUT and OUTPUT (for validation)
#define INPUT             INPUTS_PATH"1__q_activation.bin"
#define OUTPUT_CONV       INPUTS_PATH"17__tf.math.reduce_max.bin"
#define OUTPUT_GRU0       INPUTS_PATH"18__q_bidirectional.bin"
#define OUTPUT_GRU1       INPUTS_PATH"19__q_bidirectional_1.bin"
#define OUTPUT_LS         INPUTS_PATH"21__time_distributed_1.bin"
#define OUTPUT_GS         INPUTS_PATH"22__tf.math.reduce_max_1.bin"


// conv2d_0
#define CONV_0_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_kernel_merged_scale.bin"
#define CONV_0_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_kernel_scale_hls.bin"
#define CONV_0_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_bias_hls.bin"

// conv2d_1
#define CONV_1_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_1_kernel_merged_scale.bin"
#define CONV_1_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_1_kernel_scale_hls.bin"
#define CONV_1_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_1_bias_hls.bin"

// conv2d_2
#define CONV_2_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_2_kernel_merged_scale.bin"
#define CONV_2_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_2_kernel_scale_hls.bin"
#define CONV_2_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_2_bias_hls.bin"

// conv2d_3
#define CONV_3_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_3_kernel_merged_scale.bin"
#define CONV_3_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_3_kernel_scale_hls.bin"
#define CONV_3_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_3_bias_hls.bin"

// conv2d_4
#define CONV_4_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_4_kernel_merged_scale.bin"
#define CONV_4_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_4_kernel_scale_hls.bin"
#define CONV_4_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_4_bias_hls.bin"

// conv2d_5
#define CONV_5_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_5_kernel_merged_scale.bin"
#define CONV_5_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_5_kernel_scale_hls.bin"
#define CONV_5_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_5_bias_hls.bin"

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
        #ifdef DEPRECATED__READ_APFIXED
        // Calculate the number of bytes to read
        size_t numBytes = (W + 7) / 8 * arraysize;

        // Read the binary data into a temporary buffer
        char* buffer = new char[numBytes];
        fread(buffer, sizeof(char), numBytes, file);
        fclose(file);

        quant_t* p_array = (quant_t*)array;

        // Copy the binary data from the temporary buffer to the array
        size_t offset = 0;
        for (size_t i = 0; i < arraysize; ++i) {
            //printf("%4d > ", idx++);
            quant_t value = 0;
            for (size_t j = 0; j < W; ++j) {
                // Read the bits from the buffer
                bool bit = (buffer[offset / 8] >> (7 - (offset % 8))) & 1;
                //printf("%d", bit);
                // Assign the bit to the corresponding position in the value
                value[W - 1 - j] = bit;
                offset++;
            }
            //printf(" - %f\n", value.to_float());
            p_array[i] = value;
        }
        delete[] buffer;
        #else

        // Read the binary data into the array
        char* buffer = (char*)array;
        fread(buffer, typeSize, arraysize, file);
        fclose(file);
        #endif
        
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
	load(INPUT, input, IHEIGHT*IWIDTH/PACKET, sizeof(imap_t));
	load(OUTPUT_CONV, outputConv, IHEIGHT*FILTERS, sizeof(float));
	load(OUTPUT_GRU0, outputGRU0, IHEIGHT*(FILTERS*2), sizeof(float));
	load(OUTPUT_GRU1, outputGRU1, IHEIGHT*(FILTERS*2), sizeof(float));
	load(OUTPUT_LS, outputLS, IHEIGHT, sizeof(float));
	load(OUTPUT_GS, outputGS, 1, sizeof(float));
}

void loadWeights(
	weigth_t* kernel_0, weigth_t* kernel_0_scale, bias_t* bias_0,
	weigth_t* kernel_1, weigth_t* kernel_1_scale, bias_t* bias_1,
	weigth_t* kernel_2, weigth_t* kernel_2_scale, bias_t* bias_2,
	weigth_t* kernel_3, weigth_t* kernel_3_scale, bias_t* bias_3,
	weigth_t* kernel_4, weigth_t* kernel_4_scale, bias_t* bias_4,
	weigth_t* kernel_5, weigth_t* kernel_5_scale, bias_t* bias_5,
	gru_t* gru0f_kernel, gru_t* gru0f_rkernel, gru_t* gru0f_bias, gru_t* gru0f_rbias,
	gru_t* gru0b_kernel, gru_t* gru0b_rkernel, gru_t* gru0b_bias, gru_t* gru0b_rbias,
	gru_t* gru1f_kernel, gru_t* gru1f_rkernel, gru_t* gru1f_bias, gru_t* gru1f_rbias,
	gru_t* gru1b_kernel, gru_t* gru1b_rkernel, gru_t* gru1b_bias, gru_t* gru1b_rbias,
	float* td0_kernel, float* td0_bias,
	float* td1_kernel, float* td1_bias
) {
    // conv2d_0
	load(CONV_0_KERNEL, kernel_0, FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET, sizeof(weigth_t));
    load(CONV_0_KERNEL_SCALE, kernel_0_scale, CHANNELS/PACKET, sizeof(weigth_t));
    load(CONV_0_BIAS, bias_0, CHANNELS, sizeof(bias_t));

    // conv2d_1
    load(CONV_1_KERNEL, kernel_1, FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET, sizeof(weigth_t));
    load(CONV_1_KERNEL_SCALE, kernel_1_scale, CHANNELS/PACKET, sizeof(weigth_t));
    load(CONV_1_BIAS, bias_1, CHANNELS, sizeof(bias_t));

    // conv2d_2
    load(CONV_2_KERNEL, kernel_2, FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET, sizeof(weigth_t));
    load(CONV_2_KERNEL_SCALE, kernel_2_scale, CHANNELS/PACKET, sizeof(weigth_t));
    load(CONV_2_BIAS, bias_2, CHANNELS, sizeof(bias_t));
    
    // conv2d_3
    load(CONV_3_KERNEL, kernel_3, FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET, sizeof(weigth_t));
    load(CONV_3_KERNEL_SCALE, kernel_3_scale, CHANNELS/PACKET, sizeof(weigth_t));
    load(CONV_3_BIAS, bias_3, CHANNELS, sizeof(bias_t));
    
    // conv2d_4
    load(CONV_4_KERNEL, kernel_4, FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET, sizeof(weigth_t));
    load(CONV_4_KERNEL_SCALE, kernel_4_scale, CHANNELS/PACKET, sizeof(weigth_t));
    load(CONV_4_BIAS, bias_4, CHANNELS, sizeof(bias_t));
    
    // conv2d_5
    load(CONV_5_KERNEL, kernel_5, FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET, sizeof(weigth_t));
    load(CONV_5_KERNEL_SCALE, kernel_5_scale, CHANNELS/PACKET, sizeof(weigth_t));
    load(CONV_5_BIAS, bias_5, CHANNELS, sizeof(bias_t));

    // gru_0_forward
    load(GRU_0_FORWARD_KERNEL, gru0f_kernel, FILTERS*CHANNELS*GRU_SPLIT_SIZE, sizeof(gru_t));
    load(GRU_0_FORWARD_RECURRENT_KERNEL, gru0f_rkernel, FILTERS*CHANNELS*GRU_SPLIT_SIZE, sizeof(gru_t));
    load(GRU_0_FORWARD_BIAS, gru0f_bias, FILTERS*GRU_SPLIT_SIZE, sizeof(gru_t));
    load(GRU_0_FORWARD_RECURRENT_BIAS, gru0f_rbias, FILTERS*GRU_SPLIT_SIZE, sizeof(gru_t));
    // gru_0_backward
    load(GRU_0_BACKWARD_KERNEL, gru0b_kernel, FILTERS*CHANNELS*GRU_SPLIT_SIZE, sizeof(gru_t));
    load(GRU_0_BACKWARD_RECURRENT_KERNEL, gru0b_rkernel, FILTERS*CHANNELS*GRU_SPLIT_SIZE, sizeof(gru_t));
    load(GRU_0_BACKWARD_BIAS, gru0b_bias, FILTERS*GRU_SPLIT_SIZE, sizeof(gru_t));
    load(GRU_0_BACKWARD_RECURRENT_BIAS, gru0b_rbias, FILTERS*GRU_SPLIT_SIZE, sizeof(gru_t));

	// gru_1_forward
	load(GRU_1_FORWARD_KERNEL, gru1f_kernel, (FILTERS*2)*CHANNELS*GRU_SPLIT_SIZE, sizeof(gru_t));
	load(GRU_1_FORWARD_RECURRENT_KERNEL, gru1f_rkernel, FILTERS*CHANNELS*GRU_SPLIT_SIZE, sizeof(gru_t));
	load(GRU_1_FORWARD_BIAS, gru1f_bias, FILTERS*GRU_SPLIT_SIZE, sizeof(gru_t));
	load(GRU_1_FORWARD_RECURRENT_BIAS, gru1f_rbias, FILTERS*GRU_SPLIT_SIZE, sizeof(gru_t));
	// gru_1_backward
	load(GRU_1_BACKWARD_KERNEL, gru1b_kernel, (FILTERS*2)*CHANNELS*GRU_SPLIT_SIZE, sizeof(gru_t));
	load(GRU_1_BACKWARD_RECURRENT_KERNEL, gru1b_rkernel, FILTERS*CHANNELS*GRU_SPLIT_SIZE, sizeof(gru_t));
	load(GRU_1_BACKWARD_BIAS, gru1b_bias, FILTERS*GRU_SPLIT_SIZE, sizeof(gru_t));
	load(GRU_1_BACKWARD_RECURRENT_BIAS, gru1b_rbias, FILTERS*GRU_SPLIT_SIZE, sizeof(gru_t));

    // timedist_0
    load(TDIST_0_KERNEL, td0_kernel, (FILTERS*2)*FILTERS, sizeof(float));
    load(TDIST_0_BIAS, td0_bias, FILTERS, sizeof(float));

    // timedist_1
    load(TDIST_1_KERNEL, td1_kernel, FILTERS*1, sizeof(float));
    load(TDIST_1_BIAS, td1_bias, 1, sizeof(float));
}

#endif // LOAD_WEIGHTS_H
