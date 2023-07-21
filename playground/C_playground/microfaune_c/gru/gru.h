#pragma once

#ifndef GRU_H
#define GRU_H

#include "gru_settings.h"

#define GRU_SPLIT   3

#define GRU_MAX_STATE 2                                // requires to be equal to STATE_SIZE, for gru_syncState to work
extern gru_t state[GRU_MAX_STATE][GRU_STATE_SIZE];     // 0 -> current, 1 -> next


void gru_clearState();
void gru_syncState();

template
<
    int GRU_IN_COLS,
    int GRU_KERNEL_LINES,   int GRU_KERNEL_COLS,
    int GRU_KERNEL_R_LINES, int GRU_KERNEL_R_COLS,
    int GRU_BIAS_SIZE
>
void gru(
    int idx,
    const gru_t input[GRU_IN_COLS],
    const gru_t kernel[GRU_KERNEL_LINES][GRU_SPLIT][GRU_KERNEL_COLS],
    const gru_t bias[GRU_BIAS_SIZE][GRU_SPLIT],
    const gru_t recurrent_kernel[GRU_KERNEL_R_LINES][GRU_SPLIT][GRU_KERNEL_R_COLS],
    const gru_t recurrent_bias[GRU_BIAS_SIZE][GRU_SPLIT],
    gru_t output[1]
) {
    gru_t matrix_x[3];
    for (int i = 0; i < 3; ++i) {
        matrix_x[i] = 0;
        for (int j = 0; j < GRU_KERNEL_COLS; ++j) {
            gru_t iVal = input[j];
            gru_t kVal = kernel[idx][i][j];
            matrix_x[i] += iVal * kVal;
        }
    }
    for (int i = 0; i < 3; ++i) {
        matrix_x[i] += bias[idx][i];
    }


    gru_t matrix_inner[3];
    for (int i = 0; i < 3; ++i) {
        matrix_inner[i] = 0;
        for (int j = 0; j < GRU_KERNEL_R_COLS; ++j) {
            gru_t iVal = state[0][j];
            gru_t kVal = recurrent_kernel[idx][i][j];
            matrix_inner[i] += iVal * kVal;
        }
    }
    for (int i = 0; i < 3; ++i) {
        matrix_inner[i] += recurrent_bias[idx][i];
    }


    gru_t z = SIGMOID(matrix_x[0] + matrix_inner[0]);
    gru_t r = SIGMOID(matrix_x[1] + matrix_inner[1]);
    gru_t hh = TANH(matrix_x[2] + (r * matrix_inner[2]));

    gru_t out = z * state[0][idx] + (1 - z) * hh;
    state[1][idx] = out;
    output[0] = out;
}

#endif // GRU_H