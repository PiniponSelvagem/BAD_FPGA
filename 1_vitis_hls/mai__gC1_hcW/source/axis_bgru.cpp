#ifndef GRU_H
#define GRU_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include "hls_stream.h"

#include "types.h"
#include "utils.h"
#include "size_bgru.h"

/* WEIGHTS */
// GRU_0
#include "weights/gru_0_forward_bias.h"
#include "weights/gru_0_forward_bias_recurrent.h"
#include "weights/gru_0_forward_kernel.h"
#include "weights/gru_0_forward_kernel_recurrent.h"
#include "weights/gru_0_backward_bias.h"
#include "weights/gru_0_backward_bias_recurrent.h"
#include "weights/gru_0_backward_kernel.h"
#include "weights/gru_0_backward_kernel_recurrent.h"
// GRU_1
#include "weights/gru_1_forward_bias.h"
#include "weights/gru_1_forward_bias_recurrent.h"
#include "weights/gru_1_forward_kernel.h"
#include "weights/gru_1_forward_kernel_recurrent.h"
#include "weights/gru_1_backward_bias.h"
#include "weights/gru_1_backward_bias_recurrent.h"
#include "weights/gru_1_backward_kernel.h"
#include "weights/gru_1_backward_kernel_recurrent.h"
/* ******* */

typedef ap_axis<64, 0, 0, 0> in_pkt;
typedef ap_axis<64, 0, 0, 0> out_pkt;

gru_imap_t input[GRU_IN_LINES*GRU_IN_MAX_COLS];


#define GRU_MAX_STATE   2
gru_omap_t state[GRU_MAX_STATE][GRU_FILTERS];     // 0 -> current, 1 -> next

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
    }
}


//#define DEBUG_GRU
void gru_cell(
    int idx,
	int kernelCols,
    const gru_imap_t* input,
    const gru_weigth_t* kernel,    	        const gru_weigth_t* bias,
    const gru_weigth_t* recurrent_kernel,   const gru_weigth_t* recurrent_bias,
	gru_omap_t* output
) {
#ifdef DEBUG_GRU
	printf("#### STEP START ####\n");
#endif
	/*
    for (int i=0; i<kernelCols; ++i) {
        printf("iVal = %f\n", (*(input + i)).to_float());
    }
	*/

	int pkernel_offset = (idx * GRU_SPLIT_SIZE * kernelCols);
    int preckernel_offset = (idx * GRU_SPLIT_SIZE * GRU_KERNEL_REC_COLS);

    gru_matrix_t matrix_x[GRU_SPLIT_SIZE];
    GRU_cell_loop_x_row: for (int i = 0; i < GRU_SPLIT_SIZE; ++i) {
        matrix_x[i] = 0;
        int pkernel_offset_row = pkernel_offset + (i * kernelCols);
        GRU_cell_loop_x_col: for (int j = 0; j < GRU_KERNEL_COLS_MAX; ++j) {
#pragma HLS PIPELINE II=2
            if (j >= kernelCols)
                break;
            gru_imap_t iVal = *(input + j);
            gru_weigth_t kVal = *((gru_weigth_t*)kernel + pkernel_offset_row + j);
            matrix_x[i] += iVal * kVal;
            //printf("ival = %f, kval = %f\n", iVal.to_float(), kVal.to_float());
        }
    }
#ifdef DEBUG_GRU
    printf("matrix_x (dot) = [%f, %f, %f]\n", matrix_x[0].to_float(), matrix_x[1].to_float(), matrix_x[2].to_float());
#endif
    GRU_cell_loop_x_bias: for (int i = 0; i < GRU_SPLIT_SIZE; ++i) {
    	gru_weigth_t xb = *(bias + (idx * GRU_SPLIT_SIZE) + i);
        matrix_x[i] += xb;
    }
#ifdef DEBUG_GRU
    printf("matrix_x (bias_add) = [%f, %f, %f]\n", matrix_x[0].to_float(), matrix_x[1].to_float(), matrix_x[2].to_float());
#endif


    gru_matrix_t matrix_inner[GRU_SPLIT_SIZE];
    GRU_cell_loop_inner_row: for (int i = 0; i < GRU_SPLIT_SIZE; ++i) {
        matrix_inner[i] = 0;
        int preckernel_offset_row = preckernel_offset + (i * GRU_KERNEL_REC_COLS);
        GRU_cell_loop_inner_col: for (int j = 0; j < GRU_KERNEL_REC_COLS; ++j) {
        	gru_imap_t iVal = state[0][j];
        	gru_weigth_t kVal = *((gru_weigth_t*)recurrent_kernel + preckernel_offset_row + j);
            matrix_inner[i] += iVal * kVal;
        }
    }
#ifdef DEBUG_GRU
    printf("matrix_inner (dot) = [%f, %f, %f]\n", matrix_inner[0].to_float(), matrix_inner[1].to_float(), matrix_inner[2].to_float());
#endif
    GRU_cell_loop_inner_bias: for (int i = 0; i < GRU_SPLIT_SIZE; ++i) {
    	gru_weigth_t ib = *(recurrent_bias + (idx * GRU_SPLIT_SIZE) + i);
        matrix_inner[i] += ib;
    }
#ifdef DEBUG_GRU
    printf("matrix_inner (bias_add) = [%f, %f, %f]\n", matrix_inner[0].to_float(), matrix_inner[1].to_float(), matrix_inner[2].to_float());
#endif

    gru_sigmoid_t z = SIGMOID(matrix_x[0] + matrix_inner[0]);
    gru_sigmoid_t r = SIGMOID(matrix_x[1] + matrix_inner[1]);
    gru_tanh_t hh   = TANH(matrix_x[2] + (r * matrix_inner[2]));

#ifdef DEBUG_GRU
    printf("z = %f\n", z.to_float());
    printf("r = %f\n", r.to_float());
    printf("hh = %f\n", hh.to_float());
#endif

    gru_omap_t out = (z * state[0][idx]) + ((1 - z) * hh);

    state[1][idx] = out;
    *output = out;

#ifdef DEBUG_GRU
    printf("h = %f\n", out.to_float());
    printf("#### STEP END ####\n\n");
#endif
}

/**
 * layerIndex:
 * - 0 -> GRU_0_FORWARD
 * - 1 -> GRU_0_BACKWARD
 * - 2 -> GRU_1_FORWARD
 * - 3 -> GRU_1_BACKWARD
*/
void gru(
	hls::stream<in_pkt> &strm_in,
	hls::stream<in_pkt> &strm_out,
    int layerIndex
) {
#pragma HLS INTERFACE s_axilite port=return bundle=BUS1
//#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out
#pragma HLS INTERFACE s_axilite port=layerIndex bundle=BUS1

#pragma HLS ARRAY_PARTITION variable=input type=cyclic factor=4

    out_pkt tmpo;
    ap_uint<64> output = 0;
#define GRU_LEFT_TO_SEND    (64/G_IN_W_BIT_WIDTH)
    ap_uint<4> txBytesLeft = GRU_LEFT_TO_SEND;	// bytes left to fill the write transfer

    gru_weigth_t* kernel;
	gru_weigth_t* bias;
	gru_weigth_t* rkernel;
	gru_weigth_t* rbias;
	ap_uint<1> isForward;
	int kernelCols;
	int kernelSize;
	switch (layerIndex) {
		case 0:
			// GRU_0_F
			isForward = GRU_FORWARD;
			kernelCols = GRU_0__IN_COLS;
			kernelSize = GRU0_KERNEL_SIZE;
			kernel = (gru_weigth_t*)g_0_forward_kernel;
			rkernel = (gru_weigth_t*)g_0_forward_kernel_recurrent;
			bias = (gru_weigth_t*)g_0_forward_bias;
			rbias = (gru_weigth_t*)g_0_forward_bias_recurrent;
			break;
		case 1:
			// GRU_0_B
			isForward = GRU_BACKWARD;
			kernelCols = GRU_0__IN_COLS;
			kernelSize = GRU0_KERNEL_SIZE;
			kernel = (gru_weigth_t*)g_0_backward_kernel;
			rkernel = (gru_weigth_t*)g_0_backward_kernel_recurrent;
			bias = (gru_weigth_t*)g_0_backward_bias;
			rbias = (gru_weigth_t*)g_0_backward_bias_recurrent;
			break;
		case 2:
			// GRU_1_F
			isForward = GRU_FORWARD;
			kernelCols = GRU_1__IN_COLS;
			kernelSize = GRU1_KERNEL_SIZE;
			kernel = (gru_weigth_t*)g_1_forward_kernel;
			rkernel = (gru_weigth_t*)g_1_forward_kernel_recurrent;
			bias = (gru_weigth_t*)g_1_forward_bias;
			rbias = (gru_weigth_t*)g_1_forward_bias_recurrent;
			break;
		case 3:
		default:
			// GRU_1_B
			isForward = GRU_BACKWARD;
			kernelCols = GRU_1__IN_COLS;
			kernelSize = GRU1_KERNEL_SIZE;
			kernel = (gru_weigth_t*)g_1_backward_kernel;
			rkernel = (gru_weigth_t*)g_1_backward_kernel_recurrent;
			bias = (gru_weigth_t*)g_1_backward_bias;
			rbias = (gru_weigth_t*)g_1_backward_bias_recurrent;
			break;
	}

    in_pkt tmp;
	READ_INIT_MAP: for (int in = 0; in < (GRU_IN_LINES*GRU_IN_MAX_COLS); ) {
		if (in > (GRU_IN_LINES*kernelCols))
			break;
		tmp = strm_in.read();
		input[in++].range(7,0) = (int)tmp.data.range(7,0);
		input[in++].range(7,0) = (int)tmp.data.range(15,8);
		input[in++].range(7,0) = (int)tmp.data.range(23,16);
		input[in++].range(7,0) = (int)tmp.data.range(31,24);

		input[in++].range(7,0) = (int)tmp.data.range(39,32);
		input[in++].range(7,0) = (int)tmp.data.range(47,40);
		input[in++].range(7,0) = (int)tmp.data.range(55,48);
		input[in++].range(7,0) = (int)tmp.data.range(63,56);
	}

    gru_clearState();
    int row;
    int offset;
    if (isForward) { // GRU_FORWARD
        row = 0;
        offset = 0;
    }
    else { // GRU_BACKWARD
        row = GRU_IN_LINES-1;
        offset = GRU_FILTERS;
    }

    ap_uint<17> outputsLeftToSend = GRU_IN_LINES*GRU_FILTERS;	// No further calculations needed because output is 8bits width. This value does not have to be byte alligned.
    GRU_loop_row: for (int i = 0; i < GRU_IN_LINES; ++i) {
        GRU_loop_col: for (int idx = 0; idx < GRU_FILTERS; ++idx) {
        	//gru_t* input_row = input + (row * kernelCols);	/*inSize*/
            gru_imap_t* inputLine = input + i*kernelCols;

            //printf("ROW = %d\n", row);
        	//gru_omap_t* output_cell = output + (row * (GRU_FILTERS*2)) + (idx + offset);
            gru_omap_t output_cell;	// not an array because we only use 1 cell
            gru_cell(idx, kernelCols, inputLine, kernel, bias, rkernel, rbias, &output_cell);
            --outputsLeftToSend;

            // adjust packet for later send
            output = output >> G_IN_W_BIT_WIDTH;
            output.range(63,64-G_IN_W_BIT_WIDTH) = output_cell.range(G_IN_W_BIT_WIDTH-1,0);
            --txBytesLeft;

            //printf("txBytesLeft = %d | outputsLeftToSend = %d\n", txBytesLeft, outputsLeftToSend);
            if (txBytesLeft == 0) { // packet is full and ready to send
            	txBytesLeft = GRU_LEFT_TO_SEND;
                tmpo.data.range(63,0) = output.range(63,0);
                tmpo.strb = 0xFF;
                tmpo.keep = 0xFF;
                if ((isForward & (row >= GRU_IN_LINES-1)) || !isForward & (row-1 < 0))
                    tmpo.last = 1;
                else
                    tmpo.last = 0;
                //printf("LAST = %d\n", tmpo.last);
                strm_out.write(tmpo);
            }
            else if (outputsLeftToSend == 0) { // packet is not full but is last Tx, adjust and send as last
				output = output >> (G_IN_W_BIT_WIDTH*txBytesLeft);
				tmpo.data.range(63,0) = output.range(63,0);
				tmpo.strb = 0xFF;
				tmpo.keep = 0xFF;
				tmpo.last = 1;
				//printf("LAST = %d (else if)\n", tmpo.last);
				strm_out.write(tmpo);
            }
            //printf("----\n");
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

