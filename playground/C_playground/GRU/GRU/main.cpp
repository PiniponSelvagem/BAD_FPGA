#include <stdio.h>
#include <string.h>

//#include "rrnoise/gru_rnnoise.h"

#include "utils/types.h"
#include "utils/gru_settings.h"

#include "data/data_input.h"

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



gruval tf_state[128] = { 0.0 };
gruval output[128] = { 0.0 };

gruval cell_state[128] = { 0.0 };
gruval output_cell[128] = { 0.0 };

void tensorflow_gru() {
    gru_tensorflow(input, tf_state, output);

    /*
    test_sig_z_values();
    test_sig_r_values();
    test_tanh_hh_values();
    */

    // Print the output values
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int i = 0; i < INPUT_SIZE; i++) {
        gruval out = output[i];
        gruval exp = output_expected[i];
        gruval difference = out - exp;
        //printf(" %15.12f | %7.4f | %15.12f\n", out, difference, exp);
        special_print(out, difference, exp);
    }
    printf("\n");
}

void hls_gru() {
    for (int idx = 0; idx < INPUT_SIZE; ++idx) {
        gru(idx, input, cell_state, &output_cell[idx]);
    }

    printf("    OUTPUT CELL         --VS--            EXPECTED\n");
    for (int i = 0; i < INPUT_SIZE; i++) {
        gruval out = output_cell[i];
        gruval exp = output[i];
        gruval difference = out - exp;
        //printf(" %15.12f | %7.4f | %15.12f\n", out, difference, exp);
        special_print(out, difference, exp);
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
