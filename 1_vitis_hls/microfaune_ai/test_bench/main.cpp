#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../source/global_settings.h"


clock_t start_time;

void startTimer() {
    start_time = clock();
}

double stopTimer() {
    clock_t end_time = clock();
    return (double)(end_time - start_time) / CLOCKS_PER_SEC;
}

void printTimer(double elapsed_time) {
    int milliseconds = (int)((elapsed_time - (int)elapsed_time) * 1000);
    int seconds = (int)elapsed_time % 60;
    int minutes = ((int)elapsed_time / 60) % 60;
    int hours = (int)elapsed_time / 3600;

    printf("Time taken: %02d:%02d:%02d.%03d\n", hours, minutes, seconds, milliseconds);
}


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

void loadWeights();
void predict(
    const input_t input[IN_LINES][IN_COLS],
    output_t outputLS[OUT_LINES_DEBUG][OUT_COLS_DEBUG],
    output_t outputGS[OUT_SINGLE_DEBUG]
);



#include "data_input_0.h"
#include "data_input_1.h"
#include "data_input_2_no.h"
#include "data_input_3_no.h"

#include "data_out_0.h"
#include "data_out_1.h"
#include "data_out_2_no.h"
#include "data_out_3_no.h"

#define INPUT_0 0
#define INPUT_1 1
#define INPUT_2 2
#define INPUT_3 3

// SELECT INPUT
#define SELECTED_INPUT INPUT_3

#if SELECTED_INPUT == INPUT_0
    #define INPUT input_0
    #define OUTPUT_GLOBAL dataoutexp_0_global
    #define OUTPUT_LOCAL dataoutexp_0_local
#elif SELECTED_INPUT == INPUT_1
    #define INPUT input_1
    #define OUTPUT_GLOBAL dataoutexp_1_global
    #define OUTPUT_LOCAL dataoutexp_1_local
#elif SELECTED_INPUT == INPUT_2
    #define INPUT input_2
    #define OUTPUT_GLOBAL dataoutexp_2_global
    #define OUTPUT_LOCAL dataoutexp_2_local
#elif SELECTED_INPUT == INPUT_3
    #define INPUT input_3
    #define OUTPUT_GLOBAL dataoutexp_3_global
    #define OUTPUT_LOCAL dataoutexp_3_local
#else
    #error "Invalid INPUT definition"
#endif



int main() {
    #ifdef __VITIS_HLS__
    printf("VITIS HLS detected!\n");
    #endif
    #ifdef _MSC_VER
    printf("Visual Studio detected!\n");
    #endif


    loadWeights();

    output_t out_local[OUT_LINES_DEBUG][OUT_COLS_DEBUG] = { 0 };
    output_t out_global[OUT_SINGLE_DEBUG] = { 0 };

    startTimer();
    predict(INPUT, out_local, out_global);
    double elapsed_time = stopTimer();

    printf("LOCAL SCORE:\n");
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int i = 0; i < OUT_LINES_DEBUG; ++i) {
        for (int j = 0; j < OUT_COLS_DEBUG; ++j) {
            float out = out_local[i][j];
            float exp = OUTPUT_LOCAL[i][j];
            float difference = out - exp;
            special_print(out, difference, exp);
        }
    }

    printf("GLOBAL SCORE:\n");
    float out = out_global[0];
    float exp = OUTPUT_GLOBAL[0];
    float difference = out - exp;
    special_print(out, difference, exp);

    printTimer(elapsed_time);


    /*
    #define WIDTH 16 // Specify the total number of bits
    #define INT_BITS 7 // Specify the number of integer bits

    // Test values
    ap_fixed<WIDTH, INT_BITS> numbers[] = {0.5, -0.25, 0.75, 0.001, 0.01, 1.0/2/2/2/2/2/2/2/2/2};
    //numbers[5][0] = 1; // Set the lowest bit of the decimal part to 1

    int numElements = sizeof(numbers) / sizeof(numbers[0]);

    // Convert numbers to bit representation and output
    for (int i = 0; i < numElements; ++i) {
        unsigned int bits = *((unsigned int*)(&numbers[i]));

        // Display binary representation
        std::cout << "Bit representation: ";
        for (int j = WIDTH - 1; j >= 0; --j) {
            std::cout << ((bits >> j) & 1);
        }

        // Display original float value
        std::cout << " (" << numbers[i] << ")" << std::endl;
    }
    */

    return 0;
}
