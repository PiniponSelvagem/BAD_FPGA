#include "utils.h"
#include <math.h>

/*
#define DEBUG_SIGMOID
#define DEBUG_TANH
*/

typedef ap_ufixed<16,0, AP_RND, AP_SAT> normalized_t;

/* SIGMOID */

gru_sigmoid_t sigmoidTable[SIG_TABLE_SIZE];

float zsigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
void loadSigmoidTable() {
    float step = (SIG_endValue_F - SIG_startValue_F) / (SIG_TABLE_SIZE - 1);

    for (int i = 0; i < SIG_TABLE_SIZE; ++i) {
    	float x = SIG_startValue_F + i * step;
    	sigmoidTable[i] = gru_sigmoid_t(zsigmoid(x)).to_float();
    }

#ifdef DEBUG_SIGMOID
    printf("SIGMOID table:\n");
    for (int i=0; i<SIG_TABLE_SIZE; ++i) {
        printf("[%03d] %f - %f\n", i, SIG_startValue + step*i, sigmoidTable[i]);
    }
    printf("\n");
#endif
}
gru_sigmoid_t sigmoid_table(gru_matrix_t value) {
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

gru_tanh_t tanhTable[TANH_TABLE_SIZE];

void loadTanhTable() {
    float step = (TANH_endValue_F - TANH_startValue_F) / (TANH_TABLE_SIZE - 1);

    for (int i = 0; i < TANH_TABLE_SIZE; ++i) {
    	float x = TANH_startValue_F + i * step;
    	tanhTable[i] = gru_tanh_t(tanh(x)).to_float();
    }

#ifdef DEBUG_TANH
    printf("TANH table:\n");
    for (int i=0; i<TANH_TABLE_SIZE; ++i) {
        printf("[%03d] %f - %f\n", i, TANH_startValue + step*i, tanhTable[i]);
    }
    printf("\n");
#endif
}
gru_tanh_t tanh_table(gru_matrix_t value) {
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
