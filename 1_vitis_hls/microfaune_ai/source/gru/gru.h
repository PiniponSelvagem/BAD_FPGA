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
 * 
 * @note In HLS version, this should be improved.
*/
static inline gru_t stateQuant(gru_t state) {
#define OFFSET_STATE    0.125   //0.0625
#define STEP_SIZE_STATE 0.25    //0.125
#define MIN_STATE       -1
#define MAX_STATE       1

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


/**
 * @brief quantized_relu(4.0.1)
 * @param value Value to be quantized.
 * @return Quantized value.
 * @note This should be removed in HLS and replaced with ap_fixed.
*/
float gru_activation_recurrent(float value) {
#define OFFSET_AR    0.0625
#define STEP_SIZE_AR 0.125
#define MIN_AR       0.0
#define MAX_AR       1.875
    value = value + OFFSET_AR;
    if (value <= MIN_AR)
        return MIN_AR;
    else if (value >= MAX_AR)
        return MAX_AR;
    int step = int((value + 1) / STEP_SIZE_AR);
    return (step * STEP_SIZE_AR) - 1.0;
}

/**
 * @brief quantized_tahn(4)
 * @param value Value to be quantized.
 * @return Quantized value.
 * @note This should be removed in HLS and replaced with ap_fixed and a lookup table of TAHN..
*/
float gru_activation(float value) {
    value = TANH(value);
    //
#define OFFSET_A    0.0625
#define STEP_SIZE_A 0.125
#define MIN_A       -1
#define MAX_A       0.875
    value = value + OFFSET_A;
    if (value <= MIN_A)
        return MIN_A;
    else if (value >= MAX_A)
        return MAX_A;
    int step = int((value + 1) / STEP_SIZE_A);
    return (step * STEP_SIZE_A) - 1.0;
}

/**
 * @brief Converts a value to quantized 4 bits.
 * @param value: Value to be quantized.
 * @return Quantized value
 *
 * @note In HLS version, this should be removed / replaced using HLS.
*/
static inline gru_t gru_valueQuant(gru_t value) {
#define OFFSET_QUANT    0.0625
#define STEP_SIZE_QUANT 0.125
#define MIN_QUANT       -1
#define MAX_QUANT       0.875
    value = value + OFFSET_QUANT;
    if (value <= MIN_QUANT)
        return MIN_QUANT;
    else if (value >= MAX_QUANT)
        return MAX_QUANT;
    int step = int((value + 1) / STEP_SIZE_QUANT);
    return (step * STEP_SIZE_QUANT) - 1.0;
}

void gru_cell(
    gru_idx_t idx,
    gru_krl_col_t kernelCols,
    const gru_t* input,
    const gru_t* kernel,
    const gru_t* bias,
    const gru_t* recurrent_kernel,
    gru_t* output
) {
    gru_t z = 0;
    gru_t r = 0;
    gru_t h = 0;
    gru_p_kernel pkernel_offset = (idx * GRU_SPLIT * kernelCols);
    //gru_t(*pkernel)[GRU_KERNEL_COLS_MAX][GRU_SPLIT] = (gru_t(*)[GRU_KERNEL_COLS_MAX][GRU_SPLIT])kernel;
    for (gru_krl_col_t j = 0; j < GRU_KERNEL_COLS_MAX; ++j) {
        if (j >= kernelCols)
            break;
        gru_t in = input[j];
        gru_t k1 = *((gru_t*)kernel + (pkernel_offset++));
        z += in * k1; // pkernel[idx][j][0];

        gru_t k2 = *((gru_t*)kernel + (pkernel_offset++));
        r += in * k2; // pkernel[idx][j][1];

        gru_t k3 = *((gru_t*)kernel + (pkernel_offset++));
        h += in * k3; // pkernel[idx][j][2];
    }

    gru_t(*pbias)[GRU_SPLIT] = (gru_t(*)[GRU_SPLIT])bias;
    gru_t zb = pbias[idx][0];
    z += zb;
    gru_t rb = pbias[idx][1];
    r += rb;
    gru_t hb = pbias[idx][2];
    h += hb;

    gru_t rec_z = 0;
    gru_t rec_r = 0;
    gru_t rec_h = 0;
    gru_p_kernel preckernel_offset = (idx * GRU_SPLIT * GRU_KERNEL_REC_COLS);
    //gru_t(*precurrent_kernel)[GRU_UNITS][GRU_SPLIT] = (gru_t(*)[GRU_UNITS][GRU_SPLIT])recurrent_kernel;
    for (gru_krl_col_t j = 0; j < GRU_KERNEL_REC_COLS; ++j) {
        gru_t rec_z_k = *((gru_t*)recurrent_kernel + (preckernel_offset++));
        rec_z += state[0][j] * rec_z_k; // precurrent_kernel[idx][j][0];
        gru_t rec_r_k = *((gru_t*)recurrent_kernel + (preckernel_offset++));
        rec_r += state[0][j] * rec_r_k; // precurrent_kernel[idx][j][1];
        ++preckernel_offset;    // advance over 'h' kernel
    }

    z = gru_activation_recurrent(z + rec_z);
    r = gru_activation_recurrent(r + rec_r);

    preckernel_offset = (idx * GRU_SPLIT * GRU_KERNEL_REC_COLS) + 2;
    for (gru_krl_col_t j = 0; j < GRU_KERNEL_REC_COLS; ++j) {
        gru_t rec_h_k = *((gru_t*)recurrent_kernel + (preckernel_offset + 3 * j));
        rec_h += (r * state[0][j]) * rec_h_k; //precurrent_kernel[idx][j][2];
        // 3*j advance over 'z' and 'r'
    }

    gru_t hh = gru_activation(h + rec_h);

    h = z * state[0][idx] + (1 - z) * hh;

    // this was necessary because the values got too high and too low leading to -inf and +inf, and 'h' in qkeras has its range limited
    if (h > 0.875)
        h = 0.875;
    else if (h < -1)
        h = -1;

    state[1][idx] = h;    // at the time 2023-10-21, state is not quantized
    h = gru_valueQuant(h);
    *output = h;
}

void gru(
    gru_direction_t isForward,
    i128_t inCols, i128_t inSize,
    gru_krl_col_t kernelCols,
    gru_t* input,
    gru_t* kernel,    gru_t* bias,
    gru_t* recKernel,
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
            gru_t* input_row = input + (row * inSize);
            gru_t* output_cell = output + (row * RNN_COLS_GRU) + (idx + offset);
            gru_cell(idx, kernelCols, input_row, kernel, bias, recKernel, output_cell);
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

