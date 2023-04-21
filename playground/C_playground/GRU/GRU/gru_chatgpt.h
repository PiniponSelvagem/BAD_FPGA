#pragma once

// Function signature for the GRU function
void gru_chatGPT(const float input[64], const float f_kernel[64][192], const float f_r_kernel[64][192], const float f_bias[2][192],
                 const float b_kernel[64][192], const float b_r_kernel[64][192], const float b_bias[2][192], float output[128]);