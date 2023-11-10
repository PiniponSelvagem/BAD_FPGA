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




void loadWeights();
void predict(
    const input_t input[IN_LINES][IN_COLS],
    /*
    output_t outputLS[OUTPUT_LOCAL_SCORE_LINES][OUTPUT_LOCAL_SCORE_COLS],
    output_t outputGS[OUTPUT_GLOBAL_SCORE]
    */
    conv_t outarray_b[CHANNELS][IN_LINES+2][IN_COLS+2]
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
#define SELECTED_INPUT 0

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


void runModel() {
    loadWeights();
    //predict(input, outDebug);

#ifdef RUN
    output_t out_local[OUT_LINES_DEBUG][OUT_COLS_DEBUG] = { 0 };
    output_t out_global[OUT_SINGLE_DEBUG] = { 0 };

    conv_t outDebug[CHANNELS][IN_LINES+2][IN_COLS+2] = { 0 };

    startTimer();
    predict(input, /*out_local, out_global*/ outDebug);
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
#endif
}

int main() {
    runModel();

    return 0;
}
