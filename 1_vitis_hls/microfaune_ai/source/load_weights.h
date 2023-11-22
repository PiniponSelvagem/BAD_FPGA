#ifndef LOAD_WEIGHTS_H
#define LOAD_WEIGHTS_H

#include <stdio.h>

#include "types.h"
#include "size_conv3D.h"

#define INPUTS_PATH           "E:\\Rodrigo\\ISEL\\2_Mestrado\\2-ANO_1-sem\\TFM\\BAD_FPGA\\1_vitis_hls\\microfaune_ai\\source\\inputs\\"
#define WEIGHTS_PATH          "E:\\Rodrigo\\ISEL\\2_Mestrado\\2-ANO_1-sem\\TFM\\BAD_FPGA\\1_vitis_hls\\microfaune_ai\\source\\weights\\"


// DEBUG with INPUT
#define INPUT_2             INPUTS_PATH"2__q_activation.bin"


// conv2d_0
#define CONV_0_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_kernel.bin"
#define CONV_0_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_kernel_scale.bin"
#define CONV_0_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_bias.bin"

// conv2d_1
#define CONV_1_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_1_kernel.bin"
#define CONV_1_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_1_kernel_scale.bin"
#define CONV_1_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_1_bias.bin"

/*
// conv2d_2
#define CONV_2_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_2_kernel.bin"
#define CONV_2_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_2_kernel_scale.bin"
#define CONV_2_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_2_bias.bin"

// conv2d_3
#define CONV_3_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_3_kernel.bin"
#define CONV_3_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_3_kernel_scale.bin"
#define CONV_3_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_3_bias.bin"

// conv2d_4
#define CONV_4_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_4_kernel.bin"
#define CONV_4_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_4_kernel_scale.bin"
#define CONV_4_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_4_bias.bin"

// conv2d_5
#define CONV_5_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_5_kernel.bin"
#define CONV_5_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_5_kernel_scale.bin"
#define CONV_5_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_5_bias.bin"

// gru_0_forward
#define GRU_0_FORWARD_KERNEL                    WEIGHTS_PATH"q_bidirectional_gru_forward_kernel.bin"
#define GRU_0_FORWARD_KERNEL_SCALE              WEIGHTS_PATH"q_bidirectional_gru_forward_kernel_scale.bin"
#define GRU_0_FORWARD_RECURRENT_KERNEL          WEIGHTS_PATH"q_bidirectional_gru_forward_recurrent_kernel.bin"
#define GRU_0_FORWARD_RECURRENT_KERNEL_SCALE    WEIGHTS_PATH"q_bidirectional_gru_forward_recurrent_kernel_scale.bin"
#define GRU_0_FORWARD_BIAS                      WEIGHTS_PATH"q_bidirectional_gru_forward_bias.bin"
// gru_0_backward
#define GRU_0_BACKWARD_KERNEL                   WEIGHTS_PATH"q_bidirectional_gru_backward_kernel.bin"
#define GRU_0_BACKWARD_KERNEL_SCALE             WEIGHTS_PATH"q_bidirectional_gru_backward_kernel_scale.bin"
#define GRU_0_BACKWARD_RECURRENT_KERNEL         WEIGHTS_PATH"q_bidirectional_gru_backward_recurrent_kernel.bin"
#define GRU_0_BACKWARD_RECURRENT_KERNEL_SCALE   WEIGHTS_PATH"q_bidirectional_gru_backward_recurrent_kernel_scale.bin"
#define GRU_0_BACKWARD_BIAS                     WEIGHTS_PATH"q_bidirectional_gru_backward_bias.bin"

// gru_1_forward
#define GRU_1_FORWARD_KERNEL                    WEIGHTS_PATH"q_bidirectional_1_gru_forward_kernel.bin"
#define GRU_1_FORWARD_KERNEL_SCALE              WEIGHTS_PATH"q_bidirectional_1_gru_forward_kernel_scale.bin"
#define GRU_1_FORWARD_RECURRENT_KERNEL          WEIGHTS_PATH"q_bidirectional_1_gru_forward_recurrent_kernel.bin"
#define GRU_1_FORWARD_RECURRENT_KERNEL_SCALE    WEIGHTS_PATH"q_bidirectional_1_gru_forward_recurrent_kernel_scale.bin"
#define GRU_1_FORWARD_BIAS                      WEIGHTS_PATH"q_bidirectional_1_gru_forward_bias.bin"
// gru_1_backward
#define GRU_1_BACKWARD_KERNEL                   WEIGHTS_PATH"q_bidirectional_1_gru_backward_kernel.bin"
#define GRU_1_BACKWARD_KERNEL_SCALE             WEIGHTS_PATH"q_bidirectional_1_gru_backward_kernel_scale.bin"
#define GRU_1_BACKWARD_RECURRENT_KERNEL         WEIGHTS_PATH"q_bidirectional_1_gru_backward_recurrent_kernel.bin"
#define GRU_1_BACKWARD_RECURRENT_KERNEL_SCALE   WEIGHTS_PATH"q_bidirectional_1_gru_backward_recurrent_kernel_scale.bin"
#define GRU_1_BACKWARD_BIAS                     WEIGHTS_PATH"q_bidirectional_1_gru_backward_bias.bin"

// timedist_0
#define TDIST_0_KERNEL WEIGHTS_PATH"time_distributed_kernel.bin"
#define TDIST_0_BIAS   WEIGHTS_PATH"time_distributed_bias.bin"

// timedist_1
#define TDIST_1_KERNEL WEIGHTS_PATH"time_distributed_1_kernel.bin"
#define TDIST_1_BIAS   WEIGHTS_PATH"time_distributed_1_bias.bin"
*/

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

void load(const char* path, void* array, int arraysize, int typesize) {
    FILE* file = fopen(path, "rb");

    int idx = 0;
    if (file != NULL) {
        #ifdef READ_APFIXED
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
        // Calculate the number of bytes to read
        size_t numBytes = (W_BIT_WIDTH + 7) / 8 * arraysize;
        //numBytes = numBytes / FILE_READ_BYTE_MULT;	// 2023-11-21 - im not aware of the problem to why i only read half of the file, but this seems to fix it

        // Read the binary data into the array
        char* buffer = (char*)array;
        fread(buffer, sizeof(char), numBytes, file);
        fclose(file);
        #endif
        
        printf("Loaded: %s\n", path);
    }
    else {
        printf("Error loading: %s\n", path);
    }
}

void loadWeights(weigth_t* input_2, weigth_t* kernel_1, weigth_t* kernel_1_scale) {
    // debug: input
    load(INPUT_2, input_2, FILTERS*IHEIGHT*IWIDTH, sizeof(weigth_t));

    /*
    // conv2d_0
	load(CONV_0_KERNEL, kernel_0, CHANNELS*K_SIZE*K_SIZE, sizeof(quant_t));
    load(CONV_0_KERNEL_SCALE, kernel_0_scale, CHANNELS, sizeof(quant_t));
    load(CONV_0_BIAS, bias_0, CHANNELS, sizeof(quant_t));
    */

    // conv2d_1
    load(CONV_1_KERNEL, kernel_1, FILTERS*CHANNELS*K_SIZE*K_SIZE, sizeof(weigth_t));
    load(CONV_1_KERNEL_SCALE, kernel_1_scale, CHANNELS, sizeof(weigth_t));
    /*
    load(CONV_1_BIAS, bias_1, CHANNELS, sizeof(quant_t));

    PRINT_ARRAY("kernel_0", kernel_0, CHANNELS, K_SIZE, K_SIZE);
    PRINT_ARRAY("kernel_0_scale", kernel_0_scale, CHANNELS, 1, 1);
    PRINT_ARRAY("bias_0", bias_0, CHANNELS, 1, 1);
    */

    //PRINT_ARRAY_4D("kernel_1", kernel_1, K_SIZE, K_SIZE, CHANNELS, FILTERS);
    /*
    PRINT_ARRAY("kernel_1_scale", kernel_1_scale, CHANNELS, 1, 1);  // ONLY SHOWING 1st FILTER
    PRINT_ARRAY("bias_1", bias_1, CHANNELS, 1, 1);

    // conv2d_2
    load(CONV_2_KERNEL, kernel_2, 64*64*3*3, sizeof(conv_t));
    load(CONV_2_KERNEL_SCALE, kernel_2_scale, 64, sizeof(conv_t));
    load(CONV_2_BIAS, bias_2, 64, sizeof(conv_t));
    
    // conv2d_3
    load(CONV_3_KERNEL, kernel_3, 64*64*3*3, sizeof(conv_t));
    load(CONV_3_KERNEL_SCALE, kernel_3_scale, 64, sizeof(conv_t));
    load(CONV_3_BIAS, bias_3, 64, sizeof(conv_t));
    
    // conv2d_4
    load(CONV_4_KERNEL, kernel_4, 64*64*3*3, sizeof(conv_t));
    load(CONV_4_KERNEL_SCALE, kernel_4_scale, 64, sizeof(conv_t));
    load(CONV_4_BIAS, bias_4, 64, sizeof(conv_t));
    
    // conv2d_5
    load(CONV_5_KERNEL, kernel_5, 64*64*3*3, sizeof(conv_t));
    load(CONV_5_KERNEL_SCALE, kernel_5_scale, 64, sizeof(conv_t));
    load(CONV_5_BIAS, bias_5, 64, sizeof(conv_t));

    // gru_0_forward
    load(GRU_0_FORWARD_KERNEL, kernel_gru0_f, 64*64*3, sizeof(gru_t));
    load(GRU_0_FORWARD_KERNEL_SCALE, kernel_gru0_f_scale, 64*3, sizeof(gru_t));
    load(GRU_0_FORWARD_RECURRENT_KERNEL, recurrent_kernel_gru0_f, 64*64*3, sizeof(gru_t));
    load(GRU_0_FORWARD_RECURRENT_KERNEL_SCALE, recurrent_kernel_gru0_f_scale, 64*3, sizeof(gru_t));
    load(GRU_0_FORWARD_BIAS, bias_gru0_f, 64*3, sizeof(gru_t));
    // gru_0_backward
    load(GRU_0_BACKWARD_KERNEL, kernel_gru0_b, 64*64*3, sizeof(gru_t));
    load(GRU_0_BACKWARD_KERNEL_SCALE, kernel_gru0_b_scale, 64*3, sizeof(gru_t));
    load(GRU_0_BACKWARD_RECURRENT_KERNEL, recurrent_kernel_gru0_b, 64*64*3, sizeof(gru_t));
    load(GRU_0_BACKWARD_RECURRENT_KERNEL_SCALE, recurrent_kernel_gru0_b_scale, 64*3, sizeof(gru_t));
    load(GRU_0_BACKWARD_BIAS, bias_gru0_b, 64*3, sizeof(gru_t));

    // gru_1_forward
    load(GRU_1_FORWARD_KERNEL, kernel_gru1_f, 64*128*3, sizeof(gru_t));
    load(GRU_1_FORWARD_KERNEL_SCALE, kernel_gru1_f_scale, 64*3, sizeof(gru_t));
    load(GRU_1_FORWARD_RECURRENT_KERNEL, recurrent_kernel_gru1_f, 64*64*3, sizeof(gru_t));
    load(GRU_1_FORWARD_RECURRENT_KERNEL_SCALE, recurrent_kernel_gru1_f_scale, 64*3, sizeof(gru_t));
    load(GRU_1_FORWARD_BIAS, bias_gru1_f, 64*3, sizeof(gru_t));
    // gru_1_backward
    load(GRU_1_BACKWARD_KERNEL, kernel_gru1_b, 64*128*3, sizeof(gru_t));
    load(GRU_1_BACKWARD_KERNEL_SCALE, kernel_gru1_b_scale, 64*3, sizeof(gru_t));
    load(GRU_1_BACKWARD_RECURRENT_KERNEL, recurrent_kernel_gru1_b, 64*64*3, sizeof(gru_t));
    load(GRU_1_BACKWARD_RECURRENT_KERNEL_SCALE, recurrent_kernel_gru1_b_scale, 64*3, sizeof(gru_t));
    load(GRU_1_BACKWARD_BIAS, bias_gru1_b, 64*3, sizeof(gru_t));

    // timedist_0
    load(TDIST_0_KERNEL, kernel_td0, 128*64, sizeof(timedist_t));
    load(TDIST_0_BIAS, bias_td0, 64, sizeof(timedist_t));

    // timedist_1
    load(TDIST_1_KERNEL, kernel_td1, 64*1, sizeof(timedist_t));
    load(TDIST_1_BIAS, bias_td1, 1, sizeof(timedist_t));
    */
}

#endif // LOAD_WEIGHTS_H
