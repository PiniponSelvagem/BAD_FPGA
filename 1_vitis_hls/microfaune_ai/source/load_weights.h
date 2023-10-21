#ifndef LOAD_WEIGHTS_H
#define LOAD_WEIGHTS_H

//#define SYNTHESIS
#ifndef SYNTHESIS
#define W_PATH          "E:\\Rodrigo\\ISEL\\2_Mestrado\\2-ANO_1-sem\\TFM\\BAD_FPGA\\1_vitis_hls\\microfaune_ai\\test_bench\\bin_weights\\"
#ifdef USE_FLOAT
#define WEIGHTS_PATH    W_PATH"float\\"
#else
#ifdef USE_8_1
#define WEIGHTS_PATH    W_PATH"apf_8_1\\"
#else
#ifdef USE_16_8
#define WEIGHTS_PATH    W_PATH"apf_16_8\\"
#else
#define WEIGHTS_PATH    W_PATH"apf_32_8\\"
#endif
#endif
#endif


// conv2d_0
#define CONV_0_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_kernel.bin"
#define CONV_0_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_kernel_scale.bin"
#define CONV_0_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_bias.bin"

// conv2d_1
#define CONV_1_KERNEL       WEIGHTS_PATH"q_conv2d_batchnorm_1_kernel.bin"
#define CONV_1_KERNEL_SCALE WEIGHTS_PATH"q_conv2d_batchnorm_1_kernel_scale.bin"
#define CONV_1_BIAS         WEIGHTS_PATH"q_conv2d_batchnorm_1_bias.bin"

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


void load(const char* path, void* array, int arraysize, int typesize) {
    FILE* file = fopen(path, "rb");

    int idx = 0;
    if (file != NULL) {
        #ifdef USE_FLOAT
        fread(array, typesize, arraysize, file);
        fclose(file);
        #else
        // Calculate the number of bytes to read
        size_t numBytes = (W + 7) / 8 * arraysize;

        // Read the binary data into a temporary buffer
        char* buffer = new char[numBytes];
        fread(buffer, sizeof(char), numBytes, file);
        fclose(file);

        ap_fixed<W,I, AP_RND, AP_SAT>* p_array = (ap_fixed<W,I, AP_RND, AP_SAT>*)array;

        // Copy the binary data from the temporary buffer to the array
        size_t offset = 0;
        for (size_t i = 0; i < arraysize; ++i) {
            //printf("%4d > ", idx++);
            ap_fixed<W,I, AP_RND, AP_SAT> value = 0;
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
        #endif
        
        printf("Loaded: %s\n", path);
    }
    else {
        printf("Error loading: %s\n", path);
    }
}

void loadWeights() {
    /*
    load(CONV_0_BIAS, bias_0, 9, sizeof(conv_t));

    for (int i = 0; i < 9; ++i) {
        conv_t b = bias_0[i];
        printf(" %15.12f\n", b.to_float());
    }

    printf("##########\n");
    ap_fixed<32,8, AP_RND, AP_SAT> v = 0.625;
    for (int i = 0; i < 32; ++i) {
        bool bit = v[i];
        printf("%d", bit);
    }
    printf(" - %15.12f\n", v.to_float());
    for (int i = 31; i >= 0; i--) {
        bool bit = v[i];
        printf("%d", bit);
    }
    printf(" - %15.12f\n", v.to_float());

    printf("$$$$$$$$$$\n");
    #define Aw 32
    #define Ai 8
    ap_fixed<Aw,Ai, AP_RND, AP_SAT> a = -0.005151249002665281;
    //a[0] = 0;
    //a[1] = 1;
    //a[31-8] = 1;
    //a[31-9] = 1;
    //a[31-10] = 1;
    for (int i = Aw-1; i >= 0; i--) {
        bool bit = a[i];
        printf("%d", bit);
    }
    printf(" - %15.24f\n", a.to_float());
    */
    
    /*
    for (int i = 0; i < 64; i++) {
        printf("%2d:", i);
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
            	printf(" %11.8f", kernel_0[i][j][k].to_float());
            }
        }
        printf("\n");
    }
    */
    // conv2d_0
	load(CONV_0_KERNEL, kernel_0, 64*3*3, sizeof(conv_t));
    load(CONV_0_KERNEL_SCALE, kernel_0_scale, 64, sizeof(conv_t));
    load(CONV_0_BIAS, bias_0, 64, sizeof(conv_t));

    // conv2d_1
    load(CONV_1_KERNEL, kernel_1, 64*64*3*3, sizeof(conv_t));
    load(CONV_1_KERNEL_SCALE, kernel_1_scale, 64, sizeof(conv_t));
    load(CONV_1_BIAS, bias_1, 64, sizeof(conv_t));
    
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
}
#else
void loadWeights() {}
#endif // SYNTHESIS

#endif // LOAD_WEIGHTS_H
