#pragma once

#ifndef GRU_H
#define GRU_H

#include "gru_settings.h"

#ifdef __VITIS_HLS__
typedef ap_uint<1> gru_direction_t;
typedef ap_uint<7> gru_idx_t;
typedef ap_uint<2> gru_mtx_row_t;
typedef ap_uint<8> gru_krl_col_t;
typedef ap_uint<9> gru_wg_col_t;

typedef ap_uint<2> gru_state_row_t;
typedef ap_uint<7> gru_state_col_t;
#endif
#ifdef _MSC_VER
typedef int gru_direction_t;
typedef int gru_idx_t;
typedef int gru_mtx_row_t;
typedef int gru_krl_col_t;
typedef int gru_wg_col_t;

typedef int gru_state_row_t;
typedef int gru_state_col_t;
#endif


#define GRU_FORWARD  1
#define GRU_BACKWARD 0

#define GRU_INCOLS_MAX          128
#define GRU_KERNEL_LINES        64
#define GRU_KERNEL_COLS_MAX     128
#define GRU_KERNEL_REC_LINES    64
#define GRU_KERNEL_REC_COLS     64
#define GRU_BIAS_SIZE           64
#define GRU_SPLIT               3

#define GRU_STATE_SIZE  64
#define GRU_MAX_STATE   2                       // requires to be equal to STATE_SIZE, for gru_syncState to work
gru_t state[GRU_MAX_STATE][GRU_STATE_SIZE];     // 0 -> current, 1 -> next


void gru_clearState() {
    GRU_clearstate_loop_row: for (gru_state_row_t i = 0; i < GRU_MAX_STATE; ++i) {
        GRU_clearstate_loop_col: for (gru_state_col_t j = 0; j < GRU_STATE_SIZE; ++j) {
            state[i][j] = 0;
        }
    }
}

void gru_syncState() {
    GRU_syncstate_loop: for (gru_state_col_t i = 0; i < GRU_STATE_SIZE; ++i) {
        state[0][i] = state[1][i];
    }
}

    
void gru_cell(
    gru_idx_t idx,
    gru_krl_col_t kernelCols,
    const gru_t* input,
    const gru_t* kernel,
    const gru_t* bias,
    const gru_t* recurrent_kernel,
    const gru_t* recurrent_bias,
    gru_t* output
) {
    gru_t* pkernel = (gru_t*)kernel + (idx * GRU_SPLIT * kernelCols);
    gru_t* preckernel = (gru_t*)recurrent_kernel + (idx * GRU_SPLIT * GRU_KERNEL_REC_COLS);

    gru_t matrix_x[GRU_SPLIT];
    GRU_cell_loop_x_row: for (gru_mtx_row_t i = 0; i < GRU_SPLIT; ++i) {
        matrix_x[i] = 0;
        gru_t* pkernel_row = pkernel + (i * kernelCols);
        GRU_cell_loop_x_col: for (gru_krl_col_t j = 0; j < GRU_KERNEL_COLS_MAX; ++j) {
//#pragma HLS PIPELINE ii=3
            if (j >= kernelCols)
                break;
            gru_t iVal = *(input + j);
            gru_t kVal = *(pkernel_row + j);
            matrix_x[i] += iVal * kVal;
        }
    }
    GRU_cell_loop_x_bias: for (gru_mtx_row_t i = 0; i < GRU_SPLIT; ++i) {
//#pragma HLS PIPELINE ii=5
        matrix_x[i] += *(bias + (idx * GRU_SPLIT) + i);
    }


    gru_t matrix_inner[GRU_SPLIT];
    GRU_cell_loop_inner_row: for (gru_mtx_row_t i = 0; i < GRU_SPLIT; ++i) {
        matrix_inner[i] = 0;
        gru_t* preckernel_row = preckernel + (i * GRU_KERNEL_REC_COLS);
        GRU_cell_loop_inner_col: for (gru_krl_col_t j = 0; j < GRU_KERNEL_REC_COLS; ++j) {
//#pragma HLS PIPELINE ii=3
            gru_t iVal = state[0][j];
            gru_t kVal = *(preckernel_row + j);
            matrix_inner[i] += iVal * kVal;
        }
    }
    GRU_cell_loop_inner_bias: for (gru_mtx_row_t i = 0; i < GRU_SPLIT; ++i) {
//#pragma HLS PIPELINE ii=5
        matrix_inner[i] += *(recurrent_bias + (idx * GRU_SPLIT) + i);
    }


    gru_t z = (gru_t)SIGMOID(matrix_x[0] + matrix_inner[0]);
    gru_t r = (gru_t)SIGMOID(matrix_x[1] + matrix_inner[1]);
    gru_t hh = (gru_t)TANH(matrix_x[2] + (r * matrix_inner[2]));

    gru_t out = z * state[0][idx] + (1 - z) * hh;
    state[1][idx] = out;
    *output = out;
}

void gru(
    gru_direction_t isForward,
    i128_t inCols, i128_t inSize,
    gru_krl_col_t kernelCols,
    gru_t* input,
    gru_t* kernel,    gru_t* bias,
    gru_t* recKernel, gru_t* recBias,
    gru_t* output
) {
    gru_clearState();
    i512_t row;
    i64_t offset;
    if (isForward) { // GRU_FORWARD
        row = 0;
        offset = 0;
    }
    else { // GRU_BACKWARD
        row = RNN_LINES_GRU-1;
        offset = (RNN_COLS_GRU/2);
    }
    
    GRU_loop_row: for (i512_t i = 0; i < RNN_LINES_GRU; ++i) { // while(true)
        GRU_loop_col: for (i128_t idx = 0; idx < GRU_INCOLS_MAX; ++idx) {
            if (idx >= inCols)
                break;
            gru_t* input_row   = input + (row * inSize);
            gru_t* output_cell = output + (row * RNN_COLS_GRU) + (idx + offset);
            gru_cell(idx, kernelCols, input_row, kernel, bias, recKernel, recBias, output_cell);
        }

        // exit contidions and inc/dec iteration
        if (isForward) {
            ++row;
            if (row >= RNN_LINES_GRU)
                break;
        }
        else {
            --row;
            if (row < 0)
                break;
        }

        gru_syncState(); // TODO: improve by not sync
    }
}

#endif // GRU_H
