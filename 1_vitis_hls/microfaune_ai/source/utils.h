#ifndef UTILS_H
#define UTILS_H


#include <math.h>

// Taken from: https://github.com/nayayayay/sigmoid-function
#define SIGMOID_EULER_NUMBER_F 2.71828182846
float inline sigmoid(float n) {
    return (1 / (1 + powf(SIGMOID_EULER_NUMBER_F, -n)));
}

#define TANH_OFFSET      0.0625
#define TANH_STEP_SIZE   0.125
static inline float tanh_table(float input) {
    input = input + TANH_OFFSET;
    if (input <= -1)
        return -1.0;
    else if (input >= 0.875)
        return 0.875;
    int step = int((input + 1) / TANH_STEP_SIZE);
    return (step * TANH_STEP_SIZE) - 1.0;
}

#define SIGMOID(x)          sigmoid((float)(x))
#define TANH(x)             tanh((float)(x)) //tanh_table((float)(x))


#endif // UTILS_H
