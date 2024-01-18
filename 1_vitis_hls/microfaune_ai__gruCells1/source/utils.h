#ifndef UTILS_H
#define UTILS_H


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
#define SIG_TABLE_SIZE 128
extern float sigmoidTable[SIG_TABLE_SIZE];
#define SIG_startValue  -10.0
#define SIG_endValue     10.0
float zsigmoid(float x);
void loadSigmoidTable();
float sigmoid_table(float value);




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



// TANH lookup table
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#define W_TANH 8
#define I_TANH 1
#define TANH_TABLE_SIZE 256
extern float tanhTable[TANH_TABLE_SIZE];
#define TANH_startValue  -5.0
#define TANH_endValue     5.0
float ztanh(float x);
void loadTanhTable();
float tanh_table(float value);



#define SIGMOID(x)          sigmoid_table((float)(x))	//sigmoid((float)(x))
#define TANH(x)             tanh_table((float)(x)) //tanh((float)(x)) //tanh_table_qkeras((float)(x))

#endif // UTILS_H
