#ifndef LOAD_WEIGHTS_H
#define LOAD_WEIGHTS_H

//#define SYNTHESIS
#ifndef SYNTHESIS
#define W_PATH          "E:\\Rodrigo\\ISEL\\2_Mestrado\\2-ANO_1-sem\\TFM\\BAD_FPGA\\1_vitis_hls\\microfaune_ai\\test_bench\\bin_weights\\"
#ifdef USE_FLOAT
#define WEIGHTS_PATH    W_PATH"float\\"
#else
#ifdef USE_16_7
#define WEIGHTS_PATH    W_PATH"apf_16_7\\"
#define W 16
#define I 7
#else
#define WEIGHTS_PATH    W_PATH"apf_32_8\\"
#define W 32
#define I 8
#endif
#endif

// conv2d_0
#define CONV_0_KERNEL WEIGHTS_PATH"1_conv2d_kernel_0.bin"
#define CONV_0_BIAS   WEIGHTS_PATH"1_conv2d_bias_0.bin"
// bnorm_0
#define BNORM_0_GAMMA          WEIGHTS_PATH"2_batch_normalization_gamma_0.bin"
#define BNORM_0_BETA           WEIGHTS_PATH"2_batch_normalization_beta_0.bin"
#define BNORM_0_MOVINGMEAN     WEIGHTS_PATH"2_batch_normalization_moving_mean_0.bin"
#define BNORM_0_MOVINGVARIANCE WEIGHTS_PATH"2_batch_normalization_moving_variance_0.bin"

// conv2d_1
#define CONV_1_KERNEL WEIGHTS_PATH"4_conv2d_1_kernel_0.bin"
#define CONV_1_BIAS   WEIGHTS_PATH"4_conv2d_1_bias_0.bin"
// bnorm_1
#define BNORM_1_GAMMA          WEIGHTS_PATH"5_batch_normalization_1_gamma_0.bin"
#define BNORM_1_BETA           WEIGHTS_PATH"5_batch_normalization_1_beta_0.bin"
#define BNORM_1_MOVINGMEAN     WEIGHTS_PATH"5_batch_normalization_1_moving_mean_0.bin"
#define BNORM_1_MOVINGVARIANCE WEIGHTS_PATH"5_batch_normalization_1_moving_variance_0.bin"

// conv2d_2
#define CONV_2_KERNEL WEIGHTS_PATH"8_conv2d_2_kernel_0.bin"
#define CONV_2_BIAS   WEIGHTS_PATH"8_conv2d_2_bias_0.bin"
// bnorm_2
#define BNORM_2_GAMMA          WEIGHTS_PATH"9_batch_normalization_2_gamma_0.bin"
#define BNORM_2_BETA           WEIGHTS_PATH"9_batch_normalization_2_beta_0.bin"
#define BNORM_2_MOVINGMEAN     WEIGHTS_PATH"9_batch_normalization_2_moving_mean_0.bin"
#define BNORM_2_MOVINGVARIANCE WEIGHTS_PATH"9_batch_normalization_2_moving_variance_0.bin"

// conv2d_3
#define CONV_3_KERNEL WEIGHTS_PATH"11_conv2d_3_kernel_0.bin"
#define CONV_3_BIAS   WEIGHTS_PATH"11_conv2d_3_bias_0.bin"
// bnorm_3
#define BNORM_3_GAMMA          WEIGHTS_PATH"12_batch_normalization_3_gamma_0.bin"
#define BNORM_3_BETA           WEIGHTS_PATH"12_batch_normalization_3_beta_0.bin"
#define BNORM_3_MOVINGMEAN     WEIGHTS_PATH"12_batch_normalization_3_moving_mean_0.bin"
#define BNORM_3_MOVINGVARIANCE WEIGHTS_PATH"12_batch_normalization_3_moving_variance_0.bin"

// conv2d_4
#define CONV_4_KERNEL WEIGHTS_PATH"15_conv2d_4_kernel_0.bin"
#define CONV_4_BIAS   WEIGHTS_PATH"15_conv2d_4_bias_0.bin"
// bnorm_4
#define BNORM_4_GAMMA          WEIGHTS_PATH"16_batch_normalization_4_gamma_0.bin"
#define BNORM_4_BETA           WEIGHTS_PATH"16_batch_normalization_4_beta_0.bin"
#define BNORM_4_MOVINGMEAN     WEIGHTS_PATH"16_batch_normalization_4_moving_mean_0.bin"
#define BNORM_4_MOVINGVARIANCE WEIGHTS_PATH"16_batch_normalization_4_moving_variance_0.bin"

// conv2d_5
#define CONV_5_KERNEL WEIGHTS_PATH"18_conv2d_5_kernel_0.bin"
#define CONV_5_BIAS   WEIGHTS_PATH"18_conv2d_5_bias_0.bin"
// bnorm_5
#define BNORM_5_GAMMA          WEIGHTS_PATH"19_batch_normalization_5_gamma_0.bin"
#define BNORM_5_BETA           WEIGHTS_PATH"19_batch_normalization_5_beta_0.bin"
#define BNORM_5_MOVINGMEAN     WEIGHTS_PATH"19_batch_normalization_5_moving_mean_0.bin"
#define BNORM_5_MOVINGVARIANCE WEIGHTS_PATH"19_batch_normalization_5_moving_variance_0.bin"

// gru_0_forward
#define GRU_0_FORWARD_KERNEL            WEIGHTS_PATH"23_bidirectional_forward_gru_gru_cell_1_kernel_0.bin"
#define GRU_0_FORWARD_RECURRENT_KERNEL  WEIGHTS_PATH"23_bidirectional_forward_gru_gru_cell_1_recurrent_kernel_0.bin"
#define GRU_0_FORWARD_BIAS              WEIGHTS_PATH"23_bidirectional_forward_gru_gru_cell_1_bias_0.bin"
#define GRU_0_FORWARD_RECURRENT_BIAS    WEIGHTS_PATH"23_bidirectional_forward_gru_gru_cell_1_bias_0_recurrent.bin"
// gru_0_backward
#define GRU_0_BACKWARD_KERNEL            WEIGHTS_PATH"23_bidirectional_backward_gru_gru_cell_2_kernel_0.bin"
#define GRU_0_BACKWARD_RECURRENT_KERNEL  WEIGHTS_PATH"23_bidirectional_backward_gru_gru_cell_2_recurrent_kernel_0.bin"
#define GRU_0_BACKWARD_BIAS              WEIGHTS_PATH"23_bidirectional_backward_gru_gru_cell_2_bias_0.bin"
#define GRU_0_BACKWARD_RECURRENT_BIAS    WEIGHTS_PATH"23_bidirectional_backward_gru_gru_cell_2_bias_0_recurrent.bin"

// gru_1_forward
#define GRU_1_FORWARD_KERNEL            WEIGHTS_PATH"24_bidirectional_1_forward_gru_1_gru_cell_4_kernel_0.bin"
#define GRU_1_FORWARD_RECURRENT_KERNEL  WEIGHTS_PATH"24_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernel_0.bin"
#define GRU_1_FORWARD_BIAS              WEIGHTS_PATH"24_bidirectional_1_forward_gru_1_gru_cell_4_bias_0.bin"
#define GRU_1_FORWARD_RECURRENT_BIAS    WEIGHTS_PATH"24_bidirectional_1_forward_gru_1_gru_cell_4_bias_0_recurrent.bin"
// gru_1_backward
#define GRU_1_BACKWARD_KERNEL            WEIGHTS_PATH"24_bidirectional_1_backward_gru_1_gru_cell_5_kernel_0.bin"
#define GRU_1_BACKWARD_RECURRENT_KERNEL  WEIGHTS_PATH"24_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernel_0.bin"
#define GRU_1_BACKWARD_BIAS              WEIGHTS_PATH"24_bidirectional_1_backward_gru_1_gru_cell_5_bias_0.bin"
#define GRU_1_BACKWARD_RECURRENT_BIAS    WEIGHTS_PATH"24_bidirectional_1_backward_gru_1_gru_cell_5_bias_0_recurrent.bin"

// timedist_0
#define TDIST_0_KERNEL WEIGHTS_PATH"25_time_distributed_kernel_0.bin"
#define TDIST_0_BIAS   WEIGHTS_PATH"25_time_distributed_bias_0.bin"

// timedist_1
#define TDIST_1_KERNEL WEIGHTS_PATH"26_time_distributed_1_kernel_0.bin"
#define TDIST_1_BIAS   WEIGHTS_PATH"26_time_distributed_1_bias_0.bin"


void load(const char* path, void* array, int arraysize, int typesize) {
    FILE* file = fopen(path, "rb");

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

        ap_fixed<W,I>* p_array = (ap_fixed<W,I>*)array;

        // Copy the binary data from the temporary buffer to the array
        size_t offset = 0;
        for (size_t i = 0; i < arraysize; ++i) {
            ap_fixed<W,I> value = 0;
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
    ap_fixed<32,8> v = 0.625;
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
    ap_fixed<Aw,Ai> a = -0.005151249002665281;
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
    
    // conv2d_0
    load(CONV_0_KERNEL, kernel_0, 64*3*3, sizeof(conv_t));
    load(CONV_0_BIAS, bias_0, 64, sizeof(conv_t));
    // bnorm_0
    load(BNORM_0_GAMMA, gamma_0, 64, sizeof(bnorm_t));
    load(BNORM_0_BETA, beta_0, 64, sizeof(bnorm_t));
    load(BNORM_0_MOVINGMEAN, movingmean_0, 64, sizeof(bnorm_t));
    load(BNORM_0_MOVINGVARIANCE, movingvariance_0, 64, sizeof(bnorm_t));

    // conv2d_1
    load(CONV_1_KERNEL, kernel_1, 64*64*3*3, sizeof(conv_t));
    load(CONV_1_BIAS, bias_1, 64, sizeof(conv_t));
    // bnorm_1
    load(BNORM_1_GAMMA, gamma_1, 64, sizeof(bnorm_t));
    load(BNORM_1_BETA, beta_1, 64, sizeof(bnorm_t));
    load(BNORM_1_MOVINGMEAN, movingmean_1, 64, sizeof(bnorm_t));
    load(BNORM_1_MOVINGVARIANCE, movingvariance_1, 64, sizeof(bnorm_t));

    // conv2d_2
    load(CONV_2_KERNEL, kernel_2, 64*64*3*3, sizeof(conv_t));
    load(CONV_2_BIAS, bias_2, 64, sizeof(conv_t));
    // bnorm_2
    load(BNORM_2_GAMMA, gamma_2, 64, sizeof(bnorm_t));
    load(BNORM_2_BETA, beta_2, 64, sizeof(bnorm_t));
    load(BNORM_2_MOVINGMEAN, movingmean_2, 64, sizeof(bnorm_t));
    load(BNORM_2_MOVINGVARIANCE, movingvariance_2, 64, sizeof(bnorm_t));

    // conv2d_3
    load(CONV_3_KERNEL, kernel_3, 64*64*3*3, sizeof(conv_t));
    load(CONV_3_BIAS, bias_3, 64, sizeof(conv_t));
    // bnorm_3
    load(BNORM_3_GAMMA, gamma_3, 64, sizeof(bnorm_t));
    load(BNORM_3_BETA, beta_3, 64, sizeof(bnorm_t));
    load(BNORM_3_MOVINGMEAN, movingmean_3, 64, sizeof(bnorm_t));
    load(BNORM_3_MOVINGVARIANCE, movingvariance_3, 64, sizeof(bnorm_t));

    // conv2d_4
    load(CONV_4_KERNEL, kernel_4, 64*64*3*3, sizeof(conv_t));
    load(CONV_4_BIAS, bias_4, 64, sizeof(conv_t));
    // bnorm_4
    load(BNORM_4_GAMMA, gamma_4, 64, sizeof(bnorm_t));
    load(BNORM_4_BETA, beta_4, 64, sizeof(bnorm_t));
    load(BNORM_4_MOVINGMEAN, movingmean_4, 64, sizeof(bnorm_t));
    load(BNORM_4_MOVINGVARIANCE, movingvariance_4, 64, sizeof(bnorm_t));

    // conv2d_5
    load(CONV_5_KERNEL, kernel_5, 64*64*3*3, sizeof(conv_t));
    load(CONV_5_BIAS, bias_5, 64, sizeof(conv_t));
    // bnorm_5
    load(BNORM_5_GAMMA, gamma_5, 64, sizeof(bnorm_t));
    load(BNORM_5_BETA, beta_5, 64, sizeof(bnorm_t));
    load(BNORM_5_MOVINGMEAN, movingmean_5, 64, sizeof(bnorm_t));
    load(BNORM_5_MOVINGVARIANCE, movingvariance_5, 64, sizeof(bnorm_t));

    // gru_0_forward
    load(GRU_0_FORWARD_KERNEL, kernel_gru0_f, 64*3*64, sizeof(gru_t));
    load(GRU_0_FORWARD_RECURRENT_KERNEL, recurrent_kernel_gru0_f, 64*3*64, sizeof(gru_t));
    load(GRU_0_FORWARD_BIAS, bias_gru0_f, 64*3, sizeof(gru_t));
    load(GRU_0_FORWARD_RECURRENT_BIAS, recurrent_bias_gru0_f, 64*3, sizeof(gru_t));
    // gru_0_backward
    load(GRU_0_BACKWARD_KERNEL, kernel_gru0_b, 64*3*64, sizeof(gru_t));
    load(GRU_0_BACKWARD_RECURRENT_KERNEL, recurrent_kernel_gru0_b, 64*3*64, sizeof(gru_t));
    load(GRU_0_BACKWARD_BIAS, bias_gru0_b, 64*3, sizeof(gru_t));
    load(GRU_0_BACKWARD_RECURRENT_BIAS, recurrent_bias_gru0_b, 64*3, sizeof(gru_t));

    // gru_1_forward
    load(GRU_1_FORWARD_KERNEL, kernel_gru1_f, 64*3*128, sizeof(gru_t));
    load(GRU_1_FORWARD_RECURRENT_KERNEL, recurrent_kernel_gru1_f, 64*3*64, sizeof(gru_t));
    load(GRU_1_FORWARD_BIAS, bias_gru1_f, 64*3, sizeof(gru_t));
    load(GRU_1_FORWARD_RECURRENT_BIAS, recurrent_bias_gru1_f, 64*3, sizeof(gru_t));
    // gru_1_backward
    load(GRU_1_BACKWARD_KERNEL, kernel_gru1_b, 64*3*128, sizeof(gru_t));
    load(GRU_1_BACKWARD_RECURRENT_KERNEL, recurrent_kernel_gru1_b, 64*3*64, sizeof(gru_t));
    load(GRU_1_BACKWARD_BIAS, bias_gru1_b, 64*3, sizeof(gru_t));
    load(GRU_1_BACKWARD_RECURRENT_BIAS, recurrent_bias_gru1_b, 64*3, sizeof(gru_t));

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
