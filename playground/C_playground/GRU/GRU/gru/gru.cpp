#include "gru.h"

#include <stdio.h>
#include <string.h>


void gru(int idx, gruval* input, gruval* state, gruval* output) {

    gruval matrix_x[3];
    for (int i = 0; i < 3; ++i) {
        matrix_x[i] = 0;
        for (int j = 0; j < KERNEL_ROWS; ++j) {
            gruval iVal = input[j];
            gruval kVal = kernel[j][(i * 64) + idx];
            matrix_x[i] += iVal * kVal;
        }
    }
    for (int i = 0; i < 3; ++i) {
        matrix_x[i] += bias[(i * 64) + idx];
    }


    gruval matrix_inner[3];
    for (int i = 0; i < 3; ++i) {
        matrix_inner[i] = 0;
        for (int j = 0; j < KERNEL_ROWS; ++j) {
            gruval iVal = state[j];
            gruval kVal = recurrent_kernel[j][(i * 64) + idx];
            matrix_inner[i] += iVal * kVal;
        }
    }
    for (int i = 0; i < 3; ++i) {
        matrix_inner[i] += recurrent_bias[(i * 64) + idx];
    }


    gruval z = SIGMOID(matrix_x[0] + matrix_inner[0]);
    gruval r = SIGMOID(matrix_x[1] + matrix_inner[1]); 
    gruval hh = TANH(matrix_x[2] + (r * matrix_inner[2]));

    *output = z * state[idx] + (1 - z) * hh;
}