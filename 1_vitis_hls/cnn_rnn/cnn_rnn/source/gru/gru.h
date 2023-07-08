#pragma once

#ifndef GRU_H
#define GRU_H

#include "gru_settings.h"

typedef ap_uint<7> gru_idx_t;
typedef ap_uint<2> gru_mtx_row_t;
typedef ap_uint<8> gru_krl_row_t;
typedef ap_uint<9> gru_wg_col_t;

#define GRU_SPLIT   3

#define GRU_MAX_STATE 2                         // requires to be equal to STATE_SIZE, for gru_syncState to work
gru_t state[GRU_MAX_STATE][GRU_STATE_SIZE];     // 0 -> current, 1 -> next

void gru_clearState() {
    GRU_clearstate_loop_row: for (int i = 0; i < GRU_MAX_STATE; ++i) {
        GRU_clearstate_loop_col: for (int j = 0; j < GRU_STATE_SIZE; ++j) {
            state[i][j] = 0;
        }
    }
}

void gru_syncState() {
    GRU_syncstate_loop: for (int i = 0; i < GRU_STATE_SIZE; ++i) {
        state[0][i] = state[1][i];
    }
}

template
<
    int GRU_IN_COLS,
    int GRU_KERNEL_LINES,  int GRU_KERNEL_COLS,
    int GRU_KERNEL_R_LINES, int GRU_KERNEL_R_COLS,
    int GRU_BIAS_SIZE
>
void gru(
    gru_idx_t idx,
    const gru_t input[GRU_IN_COLS],
    const gru_t kernel[GRU_KERNEL_LINES][GRU_KERNEL_COLS][GRU_SPLIT],
    const gru_t bias[GRU_BIAS_SIZE][GRU_SPLIT],
    const gru_t recurrent_kernel[GRU_KERNEL_R_LINES][GRU_KERNEL_R_COLS][GRU_SPLIT],
    const gru_t recurrent_bias[GRU_BIAS_SIZE][GRU_SPLIT],
    gru_t output[1]
) {
    gru_t matrix_x[3];
    GRU_loop_x_row: for (gru_mtx_row_t i = 0; i < 3; ++i) {
        matrix_x[i] = 0;
        GRU_loop_x_col: for (gru_krl_row_t j = 0; j < GRU_KERNEL_LINES; ++j) {
#pragma HLS PIPELINE ii=3
            gru_t iVal = input[j];
            gru_t kVal = kernel[j][idx][i];
            matrix_x[i] += iVal * kVal;
        }
    }
    GRU_loop_x_bias: for (gru_mtx_row_t i = 0; i < 3; ++i) {
#pragma HLS PIPELINE ii=5
        matrix_x[i] += bias[idx][i];
    }


    gru_t matrix_inner[3];
    GRU_loop_inner_row: for (gru_mtx_row_t i = 0; i < 3; ++i) {
        matrix_inner[i] = 0;
        GRU_loop_inner_col: for (gru_krl_row_t j = 0; j < GRU_KERNEL_R_LINES; ++j) {
#pragma HLS PIPELINE ii=3
            gru_t iVal = state[0][j];
            gru_t kVal = recurrent_kernel[j][idx][i];
            matrix_inner[i] += iVal * kVal;
        }
    }
    GRU_loop_inner_bias: for (gru_mtx_row_t i = 0; i < 3; ++i) {
#pragma HLS PIPELINE ii=5
        matrix_inner[i] += recurrent_bias[idx][i];
    }


    gru_t z = (gru_t)SIGMOID((float)(matrix_x[0] + matrix_inner[0]));
    gru_t r = (gru_t)SIGMOID((float)(matrix_x[1] + matrix_inner[1]));
    gru_t hh = (gru_t)TANH((float)(matrix_x[2] + (r * matrix_inner[2])));

    gru_t out = z * state[0][idx] + (1 - z) * hh;
    state[1][idx] = out;
    output[0] = out;
}

#endif // GRU_H
