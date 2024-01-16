#include "utils.h"
#include <math.h>


/* SIGMOID */

float sigmoidTable[SIG_TABLE_SIZE];

float zsigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
void loadSigmoidTable() {
    float step = (SIG_endValue - SIG_startValue) / (SIG_TABLE_SIZE - 1);

    for (int i = 0; i < SIG_TABLE_SIZE; ++i) {
    	float x = SIG_startValue + i * step;
    	sigmoidTable[i] = ap_ufixed<W_SIG,I_SIG, AP_RND, AP_SAT>(zsigmoid(x)).to_float();
    }
}
float sigmoid_table(float value) {
    // Map x to the index in the table
    float normalizedX = (value - SIG_startValue) / (SIG_endValue - SIG_startValue);
    int index = (int)(normalizedX * (SIG_TABLE_SIZE - 1));

    index = (index < 0) ? 0 : index;
    index = (index >= SIG_TABLE_SIZE) ? SIG_TABLE_SIZE - 1 : index;

    float result = sigmoidTable[index];

    float compare = (1 / (1 + powf(2.71828182846, -value)));
    printf("SIG  -> %f | %f\n", result, compare);

    return result;
}





/* TANH */

float tanhTable[TANH_TABLE_SIZE];

void loadTanhTable() {
    float step = (TANH_endValue - TANH_startValue) / (TANH_TABLE_SIZE - 1);

    for (int i = 0; i < TANH_TABLE_SIZE; ++i) {
    	float x = TANH_startValue + i * step;
    	tanhTable[i] = ap_fixed<W_TANH,I_TANH, AP_RND, AP_SAT>(tanh(x)).to_float();
    }
}
float tanh_table(float value) {
    // Map x to the index in the table
    float normalizedX = (value - TANH_startValue) / (TANH_endValue - TANH_startValue);
    int index = (int)(normalizedX * (TANH_TABLE_SIZE - 1));

    index = (index < 0) ? 0 : index;
    index = (index >= TANH_TABLE_SIZE) ? SIG_TABLE_SIZE - 1 : index;

    float result = tanhTable[index];

    float compare = tanh(value);
    printf("TANH -> %f | %f\n", result, compare);

    return result;
}
