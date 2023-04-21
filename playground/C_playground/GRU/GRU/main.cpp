#include <stdio.h>

#include "data.h"
#include "data_simple.h"
//#include "gru_chatgpt.h"
#include "gru_rnnoise.h"


float output[128] = { 0.0 };

int main() {    
    // Call the GRU function
    /*
    gru_chatGPT(gru_input,
        forward_kernel, forward_recurrent_kernel, forward_bias,
        backward_kernel, backward_recurrent_kernel, backward_bias,
        output
    );
    */
    gru_rnnoise(output);

    // Print the output values
    printf("Output values:\n");
    for (int i = 0; i < 8; i++) {
        printf(" %.16f\n", output[i]);
    }
    printf("\n");

    return 0;
}
