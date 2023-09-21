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

typedef ap_uint<15> gru_p_kernel;
typedef ap_uint<14> gru_p_rkernel;
#endif
#ifdef _MSC_VER
typedef int gru_direction_t;
typedef int gru_idx_t;
typedef int gru_mtx_row_t;
typedef int gru_krl_col_t;
typedef int gru_wg_col_t;

typedef int gru_state_row_t;
typedef int gru_state_col_t;

typedef int gru_p_kernel;
typedef int gru_p_rkernel;
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


/**
 * @brief Converts a value to quantized 4 bits.
 * @param state: Value to be quantized.
 * @return Quantized state value
*/
static inline gru_t stateQuant(gru_t state) {
#define OFFSET_STATE    0.125   //0.0625
#define STEP_SIZE_STATE 0.25    //0.125
#define MIN_STATE       -1
#define MAX_STATE       0.75

    state = state + OFFSET_STATE;
    if (state <= MIN_STATE)
        return MIN_STATE;
    else if (state >= MAX_STATE)
        return MAX_STATE;
    int step = int((state + 1) / STEP_SIZE_STATE);
    return (step * STEP_SIZE_STATE) - 1.0;
}


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
    gru_p_kernel pkernel_offset = (idx * GRU_SPLIT * kernelCols);
    gru_p_rkernel preckernel_offset = (idx * GRU_SPLIT * GRU_KERNEL_REC_COLS);

    gru_acc_t matrix_x[GRU_SPLIT];
    GRU_cell_loop_x_row: for (gru_mtx_row_t i = 0; i < GRU_SPLIT; ++i) {
        matrix_x[i] = 0;
        gru_p_kernel pkernel_offset_row = pkernel_offset + (i * kernelCols);
        GRU_cell_loop_x_col: for (gru_krl_col_t j = 0; j < GRU_KERNEL_COLS_MAX; ++j) {
//#pragma HLS PIPELINE ii=3
            if (j >= kernelCols)
                break;
            gru_t iVal = TC(*(input + j));
            gru_t kVal = TC(*((gru_t*)kernel + pkernel_offset_row + j));
            matrix_x[i] += TC(TC(iVal) * TC(kVal));
        }
    }
    GRU_cell_loop_x_bias: for (gru_mtx_row_t i = 0; i < GRU_SPLIT; ++i) {
//#pragma HLS PIPELINE ii=5
        matrix_x[i] += TC(*(bias + (idx * GRU_SPLIT) + i));
    }


    gru_acc_t matrix_inner[GRU_SPLIT];
    GRU_cell_loop_inner_row: for (gru_mtx_row_t i = 0; i < GRU_SPLIT; ++i) {
        matrix_inner[i] = 0;
        gru_p_rkernel preckernel_offset_row = preckernel_offset + (i * GRU_KERNEL_REC_COLS);
        GRU_cell_loop_inner_col: for (gru_krl_col_t j = 0; j < GRU_KERNEL_REC_COLS; ++j) {
//#pragma HLS PIPELINE ii=3
            gru_t iVal = state[0][j];
            gru_t kVal = TC(*((gru_t*)recurrent_kernel + preckernel_offset_row + j));
            matrix_inner[i] += TC(TC(iVal) * TC(kVal));
        }
    }
    GRU_cell_loop_inner_bias: for (gru_mtx_row_t i = 0; i < GRU_SPLIT; ++i) {
//#pragma HLS PIPELINE ii=5
        matrix_inner[i] += TC(*(recurrent_bias + (idx * GRU_SPLIT) + i));
    }


    gru_acc_t z = TC((gru_t)SIGMOID(TC(TC(matrix_x[0]) + TC(matrix_inner[0]))));
    gru_acc_t r = TC((gru_t)SIGMOID(TC(TC(matrix_x[1]) + TC(matrix_inner[1]))));
    gru_acc_t hh = TC((gru_t)TANH(TC(TC(matrix_x[2]) + (TC(r * TC(matrix_inner[2]))))));    // matrix_x[2] + (r * matrix_inner[2])
    
    gru_acc_t out = TC(TC(z * state[0][idx]) + TC((1 - z) * hh));
#ifdef LOAD_ORIGINAL
    state[1][idx] = out;
#else
    state[1][idx] = stateQuant(out);
#endif
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
        GRU_loop_col: for (i128_t idx = 0; idx < 64 /*GRU_INCOLS_MAX*/; ++idx) {
            //if (idx >= inCols)
            //    break;
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





/*
void gru_cell(
    int idx 64,
    const float input[64],
    const float kernel[64][3][64],
    const float bias[64][3],
    const float recurrent_kernel[64][3][64],
    const float recurrent_bias[64][3],
    float output
) {

    float matrix_x[3];
    GRU_cell_loop_x_row: for (int i = 0; i < 3; ++i) {
        matrix_x[i] = 0;
        GRU_cell_loop_x_col: for (int j = 0; j < 64; ++j) {
            float iVal = input[j];
            float kVal = kernel[idx][i][j];
            matrix_x[i] += iVal * kVal;
        }
    }
    GRU_cell_loop_x_bias: for (int i = 0; i < 3; ++i) {
        matrix_x[i] += bias[idx][i];
    }


    float matrix_inner[3];
    GRU_cell_loop_inner_row: for (int i = 0; i < 3; ++i) {
        matrix_inner[i] = 0;
        GRU_cell_loop_inner_col: for (int j = 0; j < 64; ++j) {
            float iVal = state[0][j];
            float kVal = recurrent_kernel[idx][i][j];
            matrix_inner[i] += iVal * kVal;
        }
    }
    GRU_cell_loop_inner_bias: for (int i = 0; i < 3; ++i) {
        matrix_inner[i] += recurrent_bias[idx][i];
    }


    float z = SIGMOID(matrix_x[0] + matrix_inner[0]);
    float r = SIGMOID(matrix_x[1] + matrix_inner[1]);
    float hh = TANH(matrix_x[2] + (r * matrix_inner[2]));

    float out = (z * state[0][idx]) + ((1 - z) * hh);
    state[1][idx] = out;
    *output = out;
}
*/