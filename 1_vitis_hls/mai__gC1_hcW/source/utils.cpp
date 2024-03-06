#include "utils.h"
#include <math.h>

#include "lookup_sigmoid.h"
#include "lookup_tanh.h"

/*
#define DEBUG_SIGMOID
#define DEBUG_TANH
*/

typedef ap_ufixed<16,0, AP_RND, AP_SAT> normalized_t;

/* SIGMOID */
gru_sigmoid_t sigmoidFPGA(gru_matrix_t value) {
    // Map x to the index in the table
    normalized_t normalizedX = (value - SIG_startValue) / (SIG_endValue - SIG_startValue);
    int index = (int)(normalizedX * (SIG_TABLE_SIZE - 1));

    index = (index < 0) ? 0 : index;
    index = (index >= SIG_TABLE_SIZE) ? SIG_TABLE_SIZE - 1 : index;

    gru_sigmoid_t result = sigmoidTable[index];
#ifdef DEBUG_SIGMOID
    float expected = (1 / (1 + powf(2.71828182846, -value)));
    printf("SIG  -> %f | %f\n", result.to_float(), expected);
#endif
    return result;
}


/* TANH */
gru_tanh_t tanhFPGA(gru_matrix_t value) {
    // Map x to the index in the table
    normalized_t normalizedX = (value - TANH_startValue) / (TANH_endValue - TANH_startValue);
    int index = (int)(normalizedX * (TANH_TABLE_SIZE - 1));

    index = (index < 0) ? 0 : index;
    index = (index >= TANH_TABLE_SIZE) ? SIG_TABLE_SIZE - 1 : index;

    gru_tanh_t result = tanhTable[index];
#ifdef DEBUG_TANH
    float expected = tanh(value);
    printf("TANH -> %f | %f\n", result.to_float(), expected);
#endif
    return result;
}
