#include <stdio.h>
#include <string.h>

#include "../../source/global_settings.h"

#include "data_input_0.h"

#include "test_utils.cpp"


#define IN_LINES    431
#define IN_COLS     40
#define OUT_LINES   (IN_LINES+2)
#define OUT_COLS    (IN_COLS+2)

#define OUT_LINES_DEBUG   (IN_LINES+2)
#define OUT_COLS_DEBUG    ((IN_COLS/2)+2)


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


void predict(const input_t input[IN_LINES][IN_COLS], conv_t output[CHANNELS][OUT_LINES_DEBUG][OUT_COLS_DEBUG]);

void test_input_preconv2d(const input_t input[IN_LINES][IN_COLS], conv_t inputPad[OUT_LINES][OUT_COLS]);
void test_conv2d_0_c0(const input_t inputPad[OUT_LINES][OUT_COLS], conv_t output[CHANNELS][OUT_LINES][OUT_COLS]);



input_t test_inputPad[OUT_LINES][OUT_COLS];
void test_input_pad() {
    test_input_preconv2d(input, test_inputPad);

    for (int i = 0; i < OUT_LINES; ++i) {
        printf("%3d > ", i);
        for (int j = 0; j < OUT_COLS; ++j) {
            printf(" %10.6f", test_inputPad[i][j]);
        }
        printf("\n");
    }
}
conv_t test_conv_out[CHANNELS][OUT_LINES][OUT_COLS];
void test_conv() {
    /*
     * Input is all 0 inside function, but has values outside of it... 0.o
     */

    test_conv2d_0_c0(input_wPad, test_conv_out);

    for (int i = 0; i < OUT_LINES; ++i) {
        printf("%3d > ", i);
        for (int j = 0; j < OUT_COLS; ++j) {
            printf(" %10.6f", test_conv_out[0][i][j]);
        }
        printf("\n");
    }
}


conv_t output[CHANNELS][OUT_LINES_DEBUG][OUT_COLS_DEBUG];

#define MAX_SHOW    1
int main() {

    /*
    test_input_pad();
    test_conv();
    */

    printf("PREDICT START\n");
	predict(input, output);
    printf("PREDICT END\n");

    for (int c = 0; c < MAX_SHOW; ++c) {
        printf("[%2d] ################################################\n", c);
        for (int i = 0; i < OUT_LINES_DEBUG; ++i) {
            printf("%3d > ", i);
            for (int j = 0; j < OUT_COLS_DEBUG; ++j) {
                printf(" %12.8f", output[c][i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}
