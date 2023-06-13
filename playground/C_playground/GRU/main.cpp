#include <stdio.h>
#include <string.h>

//#include "rrnoise/gru_rnnoise.h"

#include "utils/types.h"
#include "utils/gru_settings.h"

#include "data/data_input.h"
#include "data/data_output.h"

/*
To set OPTION_X of data_tensorflow.h:
1. Solution Explorer, right-click on your project and select "Properties" from the context menu.
2. "Configuration Properties" > "C/C++" > "Preprocessor" from the left-hand side menu.
3. "Preprocessor Definitions" field, add your macro definition in the following format: OPTION_X
*/
#include "tensorflow/gru_tensorflow.h"

#include "gru/gru.h"


void special_print(gruval out, gruval difference, gruval exp) {
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


#include <math.h>
void test_tanh() {
    for (int i = 0; i < 20; ++i) {
        double in = (double)i / 10;
        double t = tanh(in);
        printf("%0.4f -> %0.18f\n", in, t);
    }
}




gruval output[OUTPUT_SIZE_OUTTER][OUTPUT_SIZE] = { 0.0 };
gruval output_cell[OUTPUT_SIZE_OUTTER][OUTPUT_SIZE] = { 0.0 };

#define OUT_MAX_PRINT 2

void tensorflow_gru() {
    gru_tf_clearState();

    for (int i=0; i<INPUT_SIZE_OUTTER; ++i)
        gru_tensorflow(input[i], output[i]);

    /*
    test_sig_z_values();
    test_sig_r_values();
    test_tanh_hh_values();
    */

    // Print the output values
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int outter = 0; outter < OUT_MAX_PRINT; ++outter) {
        printf("[%3d] ##############################################\n", outter);
        for (int i = 0; i < INPUT_SIZE; i++) {
            gruval out = output[outter][i];
            gruval exp = output_expected[outter][i];
            gruval difference = out - exp;
            //printf(" %15.12f | %7.4f | %15.12f\n", out, difference, exp);
            special_print(out, difference, exp);
        }
        printf("\n");
    }
}

void hls_gru() {
    gru_clearState();

    for (int i = 0; i < INPUT_SIZE_OUTTER; ++i) {
        for (int idx = 0; idx < INPUT_SIZE; ++idx) {
            gru(idx, input[i], &output_cell[i][idx]);
        }
        gru_syncState();
    }

    printf("    OUTPUT CELL         --VS--            EXPECTED\n");
    for (int outter = 0; outter < OUT_MAX_PRINT; ++outter) {
        printf("[%3d] ##############################################\n", outter);
        for (int i = 0; i < INPUT_SIZE; i++) {
            gruval out = output_cell[outter][i];
            gruval exp = output[outter][i];
            gruval difference = out - exp;
            //printf(" %15.12f | %7.4f | %15.12f\n", out, difference, exp);
            special_print(out, difference, exp);
        }
        printf("\n");
    }
}




int main() {
    // Call the GRU function
    //gru_rnnoise(output);
        
    tensorflow_gru();
    hls_gru();
    
    //test_tanh();
    //double value = tanh(-0.580621901964);

    return 0;
}
