#include "gru.h"

#include <stdio.h>
#include <string.h>


#define MAX_STATE 2                     // requires to be equal to STATE_SIZE, for gru_syncState to work
gruval state[MAX_STATE][STATE_SIZE];    // 0 -> current, 1 -> next

void gru_clearState() {
    for (int i = 0; i < MAX_STATE; ++i) {
        for (int j = 0; j < STATE_SIZE; ++j) {
            state[i][j] = 0;
        }
    }
}

void gru_syncState() {
    for (int i = 0; i < STATE_SIZE; ++i) {
        state[0][i] = state[1][i];
    }
}

void gru(int idx, const gruval* input, gruval* output) {

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
            gruval iVal = state[0][j];
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

    gruval out = z * state[0][idx] + (1 - z) * hh;
    state[1][idx] = out;
    *output = out;
}