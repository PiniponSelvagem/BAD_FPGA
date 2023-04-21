#include <stdio.h>
#include <math.h>

#include "gru_chatgpt.h"


void gru_chatGPT(const float input[64], const float f_kernel[64][192], const float f_r_kernel[64][192], const float f_bias[2][192],
         const float b_kernel[64][192], const float b_r_kernel[64][192], const float b_bias[2][192], float output[128]) {
    // Initialize hidden state
    float h[192] = { 0 };

    // Forward pass
    for (int i = 0; i < 192; i++) {
        float z = f_bias[0][i];
        float r = f_bias[1][i];
        for (int j = 0; j < 64; j++) {
            z += input[j] * f_kernel[j][i];
            r += input[j] * f_r_kernel[j][i];
        }
        z = 1 / (1 + expf(-z));
        r = 1 / (1 + expf(-r));

        float h_tilde = 0;
        for (int j = 0; j < 64; j++) {
            h_tilde += (r * input[j]) * f_r_kernel[j][i];
        }
        h_tilde = tanhf(h_tilde);

        h[i] = z * h_tilde + (1 - z) * h[i];
    }

    // Backward pass
    for (int i = 0; i < 192; i++) {
        float z = b_bias[0][i];
        float r = b_bias[1][i];
        for (int j = 0; j < 64; j++) {
            z += input[j] * b_kernel[j][i];
            r += input[j] * b_r_kernel[j][i];
        }
        z = 1 / (1 + expf(-z));
        r = 1 / (1 + expf(-r));

        float h_tilde = 0;
        for (int j = 0; j < 64; j++) {
            h_tilde += (r * input[j]) * b_r_kernel[j][i];
        }
        h_tilde = tanhf(h_tilde);

        h[i] = z * h_tilde + (1 - z) * h[i];
    }

    // Output
    for (int i = 0; i < 128; i++) {
        output[i] = h[i];
    }
}
