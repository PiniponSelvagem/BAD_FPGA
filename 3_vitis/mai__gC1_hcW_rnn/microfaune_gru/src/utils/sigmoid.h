#ifndef SIGMOID_H
#define SIGMOID_H

#include <math.h>

// Taken from: https://github.com/nayayayay/sigmoid-function
#define SIGMOID_EULER_NUMBER_F 2.71828182846
float inline sigmoid(float n) {
    return (1 / (1 + powf(SIGMOID_EULER_NUMBER_F, -n)));
}

#define SIGMOID_FLOAT(x)    sigmoid((float)(x))     // for TimeDistributed

#endif // SIGMOID_H
