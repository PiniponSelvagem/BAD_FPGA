#include <stdio.h>
#include <string.h>

#include "input/input.h"

#include "conv2d/conv2d.h"
#include "conv2d/data/conv2d_0.h"


void special_print(float out, float difference, float exp) {
    char difference_str[50];
    // Use snprintf to format difference as a string with 4 digits after the decimal point
    snprintf(difference_str, sizeof(difference_str), "%.12f", difference);

    // Loop through the string and replace '0' characters with '_'
    for (int i = 0; i < strlen(difference_str); i++) {
        if (difference_str[i] == '0') {
            difference_str[i] = '_';
        }
    }

    // Print the formatted string with underscores
    printf(" %15.12f | %15s | %15.12f\n", out, difference_str, exp);
}


#define OUT_MAX_PRINT   64
#include "conv2d/data/conv2d_0_outex.h"

conv_t inputpad[C2D_0__IN_LINES][C2D_0__IN_COLS] = { 0 };
conv_t outputpad[C2D_0__OUT_LINES][C2D_0__OUT_COLS] = { 0 };
conv_t output[INPUT_LINES][INPUT_COLS] = { 0 };

int main() {

    conv2d_preprocess(input, inputpad);
    conv2d<C2D_0__IN_LINES, C2D_0__IN_COLS, C2D_0__OUT_LINES, C2D_0__OUT_COLS>(inputpad, kernel_0[0], bias_0[0], outputpad);
    conv2d_postprocess(outputpad, output);

    // Print the output values
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int outter = 0; outter < OUT_MAX_PRINT; ++outter) {
        printf("[%3d] ##############################################\n", outter);
        for (int i = 0; i < INPUT_COLS; i++) {
            conv_t out = output[outter][i];
            conv_t exp = c2d_output_expected_channel0[outter][i];
            conv_t difference = out - exp;
            //printf(" %15.12f | %7.4f | %15.12f\n", out, difference, exp);
            special_print(out, difference, exp);
        }
        printf("\n");
    }

    return 0;
}
