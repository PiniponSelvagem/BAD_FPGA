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
gru_sigmoid_t sigmoidFPGA(gru_matrix_t value);




// TANH lookup table
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
gru_tanh_t tanhFPGA(gru_matrix_t value);



//#define UTILS_FLOAT

#define SIGMOID_FLOAT(x)    sigmoid((float)(x))     // for TimeDistributed
#ifdef UTILS_FLOAT
#define SIGMOID(x)          gru_sigmoid_t(sigmoid((float)x.to_float()))
#define TANH(x)             gru_tanh_t(tanh((float)x.to_float()))
#else
#define SIGMOID(x)          sigmoidFPGA(x)    //gru_sigmoid_t(sigmoid((float)x.to_float()))
#define TANH(x)             tanhFPGA(x)       //gru_tanh_t(tanh((float)x.to_float()))
#endif

#endif // UTILS_H
