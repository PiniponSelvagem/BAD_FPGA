#include <stdio.h>

//#include "rrnoise/gru_rnnoise.h"


/*
To set OPTION_X of data_tensorflow.h:
1. Solution Explorer, right-click on your project and select "Properties" from the context menu.
2. "Configuration Properties" > "C/C++" > "Preprocessor" from the left-hand side menu.
3. "Preprocessor Definitions" field, add your macro definition in the following format: OPTION_X
*/
#include "tensorflow/gru_tensorflow.h"
#include "tensorflow/data_input.h"

float state[128] = { 0.0 };     // CURRENTLY DONT CARE ABOUT STATE, ALL ZEROS
float output[128] = { 0.0 };

int main() {
    // Call the GRU function
    //gru_rnnoise(output);
    gru_tensorflow(input, state, output);

    // Print the output values
    printf("      OUTPUT           --VS--          EXPECTED\n");
    for (int i = 0; i < INPUT_SIZE; i++) {
        float out = output[i];
        float exp = output_expected[i];
        float difference = out - exp;
        printf(" %.12f | %15.12f | %.12f\n", out, difference, exp);
    }
    printf("\n");

    return 0;
}
