#ifndef GRU_H
#define GRU_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include "hls_stream.h"

#include "types.h"
#include "utils.h"
#include "size_bgru.h"


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



#define TRUNCATE_BITS
#ifdef TRUNCATE_BITS
#define Q_APF(W,I,value)		ap_fixed<W,I, AP_RND, AP_SAT>(value).to_float()
#define Q_APUF(W,I,value)		ap_ufixed<W,I, AP_RND, AP_SAT>(value).to_float()

// IN
// NOTE: IN, STATE and OUT is advised to be same size
#define W_IN 8
#define I_IN 1
// K
#define W_K 8
#define I_K 1
// MX_CALC
#define W_MX_CALC 8
#define I_MX_CALC 1
// Bias
#define W_B 8
#define I_B 1
// MXBias
#define W_B 8
#define I_B 4

// STATE
// NOTE: IN, STATE and OUT is advised to be same size
#define W_STATE 8
#define I_STATE 1
// RK
#define W_RK 8
#define I_RK 1
// MI_CALC
#define W_MI_CALC 8
#define I_MI_CALC 1
// Bias
#define W_RB 8
#define I_RB 1
// MIBias
#define W_MIB 8
#define I_MIB 4

// ZSIG_CALC
#define W_ZSIG_CALC 16
#define I_ZSIG_CALC 8
// Z	(unsigned)
#define W_Z 8
#define I_Z 0

// RSIG_CALC
#define W_RSIG_CALC 8
#define I_RSIG_CALC 4
// R	(unsigned)
#define W_R 8
#define I_R 0

// HHTANH_CALC
#define W_HHTANH_CALC 8
#define I_HHTANH_CALC 4
// HH
#define W_HH 8
#define I_HH 1

// ZSTATE_CALC
#define W_ZSTATE_CALC 8
#define I_ZSTATE_CALC 1
// ZHH_CALC
#define W_ZHH_CALC 8
#define I_ZHH_CALC 1
// OUT
// NOTE: IN, STATE and OUT is advised to be same size
#define W_OUT 8
#define I_OUT 1


// Quantization calculations
#define Q_IN(value) 		Q_APF(W_IN, I_IN, value)
#define Q_K(value) 			Q_APF(W_K, I_K, value)
#define Q_MX_CALC(value) 	Q_APF(W_MX_CALC, I_MX_CALC, value)
#define Q_B(value) 			Q_APF(W_B, I_B, value)
#define Q_MXB(value) 		Q_APF(W_B, I_B, value)

#define Q_STATE(value) 		Q_APF(W_STATE, I_STATE, value)
#define Q_RK(value) 		Q_APF(W_RK, I_RK, value)
#define Q_MI_CALC(value) 	Q_APF(W_MI_CALC, I_MI_CALC, value)
#define Q_RB(value)			Q_APF(W_RB, I_RB, value)
#define Q_MIB(value) 		Q_APF(W_MIB, I_MIB, value)

#define Q_ZSIG_CALC(value) 	Q_APF(W_ZSIG_CALC, I_ZSIG_CALC, value)
#define Q_Z(value) 			Q_APUF(W_Z, I_Z, value)

#define Q_RSIG_CALC(value) 	Q_APF(W_RSIG_CALC, I_RSIG_CALC, value)
#define Q_R(value) 			Q_APUF(W_R, I_R, value)

#define Q_HHTANH_CALC(value) 	Q_APF(W_HHTANH_CALC, I_HHTANH_CALC, value)
#define Q_HH(value) 			Q_APF(W_HH, I_HH, value)

#define Q_ZSTATE_CALC(value) 	Q_APF(W_ZSTATE_CALC, I_ZSTATE_CALC, value)
#define Q_ZHH_CALC(value) 		Q_APF(W_ZHH_CALC, I_ZHH_CALC, value)
#define Q_OUT(value) 			Q_APF(W_OUT, I_OUT, value)

#else

#define Q_IN(x) x
#define Q_K(x) x
#define Q_MX_CALC(x) x
#define Q_B(x) x
#define Q_MXB(x) x

#define Q_STATE(x) x
#define Q_RK(x) x
#define Q_MI_CALC(x) x
#define Q_RB(x) x
#define Q_MIB(x) x

#define Q_ZSIG_CALC(x) x
#define Q_Z(x) x

#define Q_RSIG_CALC(x) x
#define Q_R(x) x

#define Q_HHTANH_CALC(x) x
#define Q_HH(x) x

#define Q_ZSTATE_CALC(x) x
#define Q_ZHH_CALC(x) x
#define Q_OUT(x) x

#endif // TRUNCATE_BITS


//#define STATISTICS
/*
in, k, b,
m0,m1,m2, 		mb0,mb1,mb2,

state, rk, rb,
mi0,mi1,mi2, 	mib0,mib1,mib2

zsig,z,
rsig,r,
hhtanh,hh,
zstate,zhh,out
*/
#define STATS_SIZE	27
float max[STATS_SIZE] = {0};
float min[STATS_SIZE] = {0};

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
            matrix_x[i] += Q_MX_CALC((Q_IN(iVal) * Q_K(kVal)));
#ifdef STATISTICS
            // STATISTICS
            if (max[0] < iVal) max[0] = iVal;
            if (min[0] > iVal) min[0] = iVal;
            if (max[1] < kVal) max[1] = kVal;
            if (min[1] > kVal) min[1] = kVal;
			if (max[i+3] < iVal * kVal) max[i+3] = iVal * kVal;
			if (min[i+3] > iVal * kVal) min[i+3] = iVal * kVal;
#endif //!STATISTICS
        }
    }
#ifdef DEBUG_GRU
    printf("matrix_x (dot) = [%f, %f, %f]\n", matrix_x[0], matrix_x[1], matrix_x[2]);
#endif
    GRU_cell_loop_x_bias: for (int i = 0; i < GRU_SPLIT_SIZE; ++i) {
    	gru_t xb = Q_B(*(bias + (idx * GRU_SPLIT_SIZE) + i));
        matrix_x[i] += xb;
        matrix_x[i] = Q_MXB(matrix_x[i]);
#ifdef STATISTICS
        // STATISTICS
        if (max[2] < xb) max[2] = xb;
        if (min[2] > xb) min[2] = xb;
		if (max[i+6] < matrix_x[i]) max[i+6] = matrix_x[i];
		if (min[i+6] > matrix_x[i]) min[i+6] = matrix_x[i];
#endif //!STATISTICS
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
            matrix_inner[i] += Q_MI_CALC((Q_STATE(iVal) * Q_RK(kVal)));

#ifdef STATISTICS
            // STATISTICS
            if (max[9] < iVal) max[9] = iVal;
            if (min[9] > iVal) min[9] = iVal;
            if (max[10] < kVal) max[10] = kVal;
            if (min[10] > kVal) min[10] = kVal;
			if (max[i+12] < iVal * kVal) max[i+12] = iVal * kVal;
			if (min[i+12] > iVal * kVal) min[i+12] = iVal * kVal;
#endif //!STATISTICS
        }
    }
#ifdef DEBUG_GRU
    printf("matrix_inner (dot) = [%f, %f, %f]\n", matrix_inner[0], matrix_inner[1], matrix_inner[2]);
#endif
    GRU_cell_loop_inner_bias: for (int i = 0; i < GRU_SPLIT_SIZE; ++i) {
    	gru_t ib = Q_RB(*(recurrent_bias + (idx * GRU_SPLIT_SIZE) + i));
        matrix_inner[i] += ib;
        matrix_inner[i] = Q_MIB(matrix_inner[i]);
#ifdef STATISTICS
        // STATISTICS
        if (max[11] < ib) max[11] = ib;
        if (min[11] > ib) min[11] = ib;
		if (max[i+15] < matrix_inner[i]) max[i+15] = matrix_inner[i];
		if (min[i+15] > matrix_inner[i]) min[i+15] = matrix_inner[i];
#endif //!STATISTICS
    }
#ifdef DEBUG_GRU
    printf("matrix_inner (bias_add) = [%f, %f, %f]\n", matrix_inner[0], matrix_inner[1], matrix_inner[2]);
#endif

    gru_t z = Q_Z((gru_t)SIGMOID(Q_ZSIG_CALC(matrix_x[0] + matrix_inner[0])));
    gru_t r = Q_R((gru_t)SIGMOID(Q_RSIG_CALC(matrix_x[1] + matrix_inner[1])));
    gru_t hh = Q_HH((gru_t)TANH(Q_HHTANH_CALC(matrix_x[2] + (r * matrix_inner[2]))));

#ifdef STATISTICS
    // STATISTICS
    //zsig
    if (max[18] < matrix_x[0] + matrix_inner[0])	max[18] = matrix_x[0] + matrix_inner[0];
    if (min[18] > matrix_x[0] + matrix_inner[0])	min[18] = matrix_x[0] + matrix_inner[0];
    //z
    if (max[19] < z)	max[19] = z;
    if (min[19] > z)	min[19] = z;

    //rsig
    if (max[20] < matrix_x[1] + matrix_inner[1])	max[20] = matrix_x[1] + matrix_inner[1];
    if (min[20] > matrix_x[1] + matrix_inner[1])	min[20] = matrix_x[1] + matrix_inner[1];
    //r
	if (max[21] < r)	max[21] = r;
	if (min[21] > r)	min[21] = r;

    //hhtanh
	if (max[22] < matrix_x[2] + (r * matrix_inner[2]))	max[22] = matrix_x[2] + (r * matrix_inner[2]);
	if (min[22] > matrix_x[2] + (r * matrix_inner[2]))	min[22] = matrix_x[2] + (r * matrix_inner[2]);
    //hh
	if (max[23] < hh)	max[23] = hh;
	if (min[23] > hh)	min[23] = hh;
#endif //!STATISTICS

#ifdef DEBUG_GRU
    printf("z = %f\n", z);
    printf("r = %f\n", r);
    printf("hh = %f\n", hh);
#endif

    gru_t out = Q_OUT(((Q_ZSTATE_CALC(z * state[0][idx])) + Q_ZHH_CALC(((1 - z) * hh))));

#ifdef STATISTICS
    // STATISTICS
    //zstate
    if (max[24] < z * state[0][idx])	max[24] = z * state[0][idx];
    if (min[24] > z * state[0][idx])	min[24] = z * state[0][idx];
    //zhh
    if (max[25] < (1 - z) * hh)	max[25] = (1 - z) * hh;
    if (min[25] > (1 - z) * hh)	min[25] = (1 - z) * hh;
	//out
    if (max[26] < out)	max[26] = out;
    if (min[26] > out)	min[26] = out;
#endif //!STATISTICS

    state[1][idx] = out;
    *output = out;
#ifdef DEBUG_GRU
    printf("h = %f\n", out);
    printf("#### STEP END ####\n\n");
#endif
}

void printStats() {
#ifdef STATISTICS
	printf("STATS: (only the last statistics matter)\n");
    printf("MAX -> ");
    for (int i=0; i<STATS_SIZE; ++i) {
    	printf("%f ", max[i]);
    }
    printf("\n");

    printf("MIN -> ");
    for (int i=0; i<STATS_SIZE; ++i) {
    	printf("%f ", min[i]);
    }
    printf("\n");
#endif //!STATISTICS
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

    printStats();	// TODO: remove later
}

#endif // GRU_H

