#include <stdio.h>
#include <string.h>

#include "../../source/global_settings.h"

#include "data_input_0.h"

#include "test_utils.cpp"


#define IN_LINES    431
#define IN_COLS     40
#define OUT_LINES   (IN_LINES+2)
#define OUT_COLS    (IN_COLS+2)

#define OUT_LINES_DEBUG   431
#define OUT_COLS_DEBUG    1
#define OUT_SINGLE_DEBUG  1


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


void predict(
    const input_t input[IN_LINES][IN_COLS],
    output_t outputLS[OUT_LINES_DEBUG][OUT_COLS_DEBUG],
    output_t outputGS[OUT_SINGLE_DEBUG]
);

/*
void test_input_preconv2d(const input_t input[IN_LINES][IN_COLS], input_t inputPad[OUT_LINES][OUT_COLS]);
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
     *

    test_conv2d_0_c0(input_wPad, test_conv_out);

    for (int i = 0; i < OUT_LINES; ++i) {
        printf("%3d > ", i);
        for (int j = 0; j < OUT_COLS; ++j) {
            printf(" %10.6f", test_conv_out[0][i][j]);
        }
        printf("\n");
    }
}
*/

#include "z_outputexpected/dataout_0.h"
int main() {

    /*
    input_t inputpad[433][42] = { 0 };
    predict(input, inputpad);

    for (int i = 0; i < OUT_LINES; ++i) {
        printf("%3d > ", i);
        for (int j = 0; j < OUT_COLS; ++j) {
            printf(" %10.6f", inputpad[i][j].to_float());
        }
        printf("\n");
    }
    */

    output_t out_local[OUT_LINES_DEBUG][OUT_COLS_DEBUG] = { 0 };
    output_t out_global[OUT_SINGLE_DEBUG] = { 0 };

    predict(input, out_local, out_global);
    
    printf("LOCAL SCORE:\n");
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int i = 0; i < OUT_LINES_DEBUG; ++i) {
        for (int j = 0; j < OUT_COLS_DEBUG; ++j) {
            float out = out_local[i][j];
            float exp = dataoutexp_0_local[i][j];
            float difference = out - exp;
            special_print(out, difference, exp);
        }
    }

    printf("GLOBAL SCORE:\n");
    float out = out_global[0];
    float exp = dataoutexp_0_global[0];
    float difference = out - exp;
    special_print(out, difference, exp);
    return 0;
}



/*

Para semana de 10 Julho:
- excel com nº de ciclos de cada layer, cada passo, etc
- excel com estatisticas do HLS, ainda com float (baseline)
- organizar melhor a estrutura dos pesos das GRU
- mudar para ap_fixed 8 bits e ver as melhorias, meter no excel, escrever algum texto

*/
