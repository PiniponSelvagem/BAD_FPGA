#ifndef UTILS_H
#define UTILS_H

#include "types.h"

#include <math.h>

// Taken from: https://github.com/nayayayay/sigmoid-function
#define SIGMOID_EULER_NUMBER_F 2.71828182846
float inline sigmoid(float n) {
    return (1 / (1 + powf(SIGMOID_EULER_NUMBER_F, -n)));
}




// SIGMOID lookup table
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#define W_SIG 8
#define I_SIG 0
#define SIG_TABLE_SIZE 256
extern gru_sigmoid_t sigmoidTable[SIG_TABLE_SIZE];
#define SIG_startValue_F    -6.3
#define SIG_endValue_F      5.2
#define SIG_startValue      gru_matrix_t(SIG_startValue_F)
#define SIG_endValue        gru_matrix_t(SIG_endValue_F)
float zsigmoid(float x);
void loadSigmoidTable();
gru_sigmoid_t sigmoid_table(gru_matrix_t value);




/*
#define TANH_OFFSET      0.0625
#define TANH_STEP_SIZE   0.125
static inline float tanh_table_qkeras(float input) {
    input = input + TANH_OFFSET;
    if (input <= -1)
        return -1.0;
    else if (input >= 0.875)
        return 0.875;
    int step = int((input + 1) / TANH_STEP_SIZE);
    return (step * TANH_STEP_SIZE) - 1.0;
}
*/



// TANH lookup table
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#define W_TANH 8
#define I_TANH 1
#define TANH_TABLE_SIZE 256
extern gru_tanh_t tanhTable[TANH_TABLE_SIZE];
#define TANH_startValue_F   -2.6
#define TANH_endValue_F     2.6
#define TANH_startValue     gru_matrix_t(TANH_startValue_F)
#define TANH_endValue       gru_matrix_t(TANH_endValue_F)
float ztanh(float x);
void loadTanhTable();
gru_tanh_t tanh_table(gru_matrix_t value);



#define UTILS_FLOAT

#define SIGMOID_FLOAT(x)    sigmoid((float)(x))
#ifdef UTILS_FLOAT
#define SIGMOID(x)          gru_sigmoid_t(sigmoid((float)x.to_float()))
#define TANH(x)             gru_tanh_t(tanh((float)x.to_float()))
#else
#define SIGMOID(x)          sigmoid_table(x)    //gru_sigmoid_t(sigmoid((float)x.to_float()))
#define TANH(x)             tanh_table(x)       //gru_tanh_t(tanh((float)x.to_float()))
#endif

#endif // UTILS_H
