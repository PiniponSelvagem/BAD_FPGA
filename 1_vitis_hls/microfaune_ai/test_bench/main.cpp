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




#define INPUT_BIRD_0 0      // expected: 0.98303944
#define INPUT_BIRD_1 1      // expected: 0.8675719
#define INPUT_BIRD_2 2      // expected: 0.921088
#define INPUT_BIRD_3 3      // expected: 0.9579106

#define INPUT_NO_BIRD_0 4   // expected: 0.04347346
#define INPUT_NO_BIRD_1 5   // expected: 0.05368349
#define INPUT_NO_BIRD_2 6   // expected: 0.07247392
#define INPUT_NO_BIRD_3 7   // expected: 0.11154108

// SELECT INPUT
#define SELECTED_INPUT 1

#if SELECTED_INPUT == INPUT_BIRD_0
    #include "data_bird_0.h"
#elif SELECTED_INPUT == INPUT_BIRD_1
    #include "data_bird_1.h"
#elif SELECTED_INPUT == INPUT_BIRD_2
    #include "data_bird_2.h"
#elif SELECTED_INPUT == INPUT_BIRD_3
    #include "data_bird_3.h"
#elif SELECTED_INPUT == INPUT_NO_BIRD_0
    #include "data_no_bird_0.h"
#elif SELECTED_INPUT == INPUT_NO_BIRD_1
    #include "data_no_bird_1.h"
#elif SELECTED_INPUT == INPUT_NO_BIRD_2
    #include "data_no_bird_2.h"
#elif SELECTED_INPUT == INPUT_NO_BIRD_3
    #include "data_no_bird_3.h"
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
    predict(input, out_local, out_global);
    double elapsed_time = stopTimer();

    printf("LOCAL SCORE:\n");
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int i = 0; i < OUT_LINES_DEBUG; ++i) {
        for (int j = 0; j < OUT_COLS_DEBUG; ++j) {
            float out = out_local[i][j];
            float exp = dataoutexp_local[i][j];
            float difference = out - exp;
            special_print(out, difference, exp);
        }
    }

    printf("GLOBAL SCORE:\n");
    float out = out_global[0];
    float exp = dataoutexp_global[0];
    float difference = out - exp;
    special_print(out, difference, exp);

    printTimer(elapsed_time);

    /*
    #define WIDTH 8 // Specify the total number of bits
    #define INT_BITS 1 // Specify the number of integer bits

    // Test values
    ap_fixed<WIDTH, INT_BITS, AP_RND, AP_SAT> numbers[] = {
    		-0.109375,
    		-0.09375,
    		0.0625,
    		0.015625,
    		-0.078125,
    		0.0,
    		-0.0625,
    		0.03125,
    		0.046875,

    		-0.01171875,
    		-0.001953125,

			10,
			0.5,
			-10,
			0.5,
			0.000001,
			0.5,
			-0.000001
    };
    //ap_fixed<WIDTH, INT_BITS> numbers[] = {-0.109375, -0.09375, 0.0625, 0.015625, -0.078125, 0.0, -0.0625, 0.03125, 0.046875};
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
