#ifndef GRU_H
#define GRU_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include "hls_stream.h"

#include "types.h"
#include "size_bgru.h"

////////////////////////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////////////





#define GRU_MAX_STATE   2
gru_t state[GRU_MAX_STATE][GRU_FILTERS];     // 0 -> current, 1 -> next

void gru_clearState() {
    GRU_clearstate_loop_row: for (int i = 0; i < GRU_MAX_STATE; ++i) {
        GRU_clearstate_loop_col: for (int j = 0; j < GRU_FILTERS; ++j) {
            state[i][j] = 0;
        }
    }
}

void gru_syncState() {
    ///printf("STATE:");
    GRU_syncstate_loop: for (int i = 0; i < GRU_FILTERS; ++i) {
        state[0][i] = state[1][i];
        //printf("%f\n", state[0][i]);
    }
}



//#define DEBUG_GRU

void gru_cell(
    int idx,
	int kernelCols,
    const gru_t* input,
    const gru_t* kernel,    		const gru_t* bias,
    const gru_t* recurrent_kernel,  const gru_t* recurrent_bias,
	gru_t* output
) {
#ifdef DEBUG_GRU
	printf("#### STEP START ####\n");
#endif
	int pkernel_offset = (idx * GRU_SPLIT_SIZE * kernelCols);
    int preckernel_offset = (idx * GRU_SPLIT_SIZE * GRU_KERNEL_REC_COLS);

    gru_t matrix_x[GRU_SPLIT_SIZE];
    GRU_cell_loop_x_row: for (int i = 0; i < GRU_SPLIT_SIZE; ++i) {
        matrix_x[i] = 0;
        int pkernel_offset_row = pkernel_offset + (i * kernelCols);
        GRU_cell_loop_x_col: for (int j = 0; j < GRU_KERNEL_COLS_MAX; ++j) {
            if (j >= kernelCols)
                break;
            gru_t iVal = *(input + j);
            gru_t kVal = *((gru_t*)kernel + pkernel_offset_row + j);
            //printf("*iVal = %f, *kVal = %f\n", iVal, kVal);
            matrix_x[i] += iVal * kVal;
        }
    }
#ifdef DEBUG_GRU
    printf("matrix_x (dot) = [%f, %f, %f]\n", matrix_x[0], matrix_x[1], matrix_x[2]);
#endif
    GRU_cell_loop_x_bias: for (int i = 0; i < GRU_SPLIT_SIZE; ++i) {
        matrix_x[i] += *(bias + (idx * GRU_SPLIT_SIZE) + i);
    }
#ifdef DEBUG_GRU
    printf("matrix_x (bias_add) = [%f, %f, %f]\n", matrix_x[0], matrix_x[1], matrix_x[2]);
#endif


    gru_t matrix_inner[GRU_SPLIT_SIZE];
    GRU_cell_loop_inner_row: for (int i = 0; i < GRU_SPLIT_SIZE; ++i) {
        matrix_inner[i] = 0;
        int preckernel_offset_row = preckernel_offset + (i * GRU_KERNEL_REC_COLS);
        GRU_cell_loop_inner_col: for (int j = 0; j < GRU_KERNEL_REC_COLS; ++j) {
        	gru_t iVal = state[0][j];
        	gru_t kVal = *((gru_t*)recurrent_kernel + preckernel_offset_row + j);
        	//printf("*iVal = %f, *kVal = %f\n", iVal, kVal);
            matrix_inner[i] += iVal * kVal;
        }
    }
#ifdef DEBUG_GRU
    printf("matrix_inner (dot) = [%f, %f, %f]\n", matrix_inner[0], matrix_inner[1], matrix_inner[2]);
#endif
    GRU_cell_loop_inner_bias: for (int i = 0; i < GRU_SPLIT_SIZE; ++i) {
        matrix_inner[i] += *(recurrent_bias + (idx * GRU_SPLIT_SIZE) + i);
    }
#ifdef DEBUG_GRU
    printf("matrix_inner (bias_add) = [%f, %f, %f]\n", matrix_inner[0], matrix_inner[1], matrix_inner[2]);
#endif


    gru_t z = (gru_t)SIGMOID(matrix_x[0] + matrix_inner[0]);
    gru_t r = (gru_t)SIGMOID(matrix_x[1] + matrix_inner[1]);
    gru_t hh = (gru_t)TANH(matrix_x[2] + (r * matrix_inner[2]));
#ifdef DEBUG_GRU
    printf("z = %f\n", z);
    printf("r = %f\n", r);
    printf("hh = %f\n", hh);
#endif

    gru_t out = (z * state[0][idx]) + ((1 - z) * hh);
    state[1][idx] = out;
    *output = out;
#ifdef DEBUG_GRU
    printf("h = %f\n", out);
    printf("#### STEP END ####\n\n");
#endif
}

void gru(
    int isForward,
    int kernelCols,
	gru_t* input,
	gru_t* kernel,    gru_t* bias,
	gru_t* recKernel, gru_t* recBias,
	gru_t* output
) {
    gru_clearState();
    int row;
    int offset;
    if (isForward) { // GRU_FORWARD
        row = 0;
        offset = 0;
    }
    else { // GRU_BACKWARD
        row = GRU_IN_LINES-1;
        offset = GRU_FILTERS;	//(TG_GRU_IN_COLS/2);
    }

    GRU_loop_row: for (int i = 0; i < GRU_IN_LINES; ++i) { // while(true)
        GRU_loop_col: for (int idx = 0; idx < GRU_FILTERS; ++idx) {
            //if (idx >= inCols)
            //    break;
        	gru_t* input_row = input + (row * kernelCols);	/*inSize*/
        	gru_t* output_cell = output + (row * (GRU_FILTERS*2)) + (idx + offset);
            gru_cell(idx, kernelCols, input_row, kernel, bias, recKernel, recBias, output_cell);
        }
        // exit contidions and inc/dec iteration
        if (isForward) {
            ++row;
            if (row >= GRU_IN_LINES)
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

