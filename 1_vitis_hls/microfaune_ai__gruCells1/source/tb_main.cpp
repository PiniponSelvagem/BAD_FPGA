#define HW_IP 1
#include "hls_stream.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>

#include "types.h"
#include "size_conv3D.h"
#include "size_bgru.h"

#include "loader.h"

#include "soft_timedist.h"
#include "soft_reducemax.h"

#include "utils.h"


//#define PRINT_STATS
//#define DEBUG_CONV
#define DO_CONV
#define VALIDATE_OUTPUT


typedef ap_axis<64, 0, 0, 0> in_pkt;
typedef ap_axis<64, 0, 0, 0> out_pkt;

hls::stream<in_pkt> str_in;
hls::stream<out_pkt> str_out;


void conv2D(hls::stream<in_pkt> &strm_in, hls::stream<out_pkt> &strm_out, int pool, int maxWidth);
void gru(
	hls::stream<in_pkt> &strm_in,
	int isForward,
	int kernelCols,
	int kernelSize,
	gru_omap_t* output
);

imap_t input_0[IHEIGHT*IWIDTH/PACKET_CNN];
imap_t input_1[IHEIGHT*IWIDTH*CHANNELS/PACKET_CNN];
imap_t input_2[IHEIGHT*IWIDTH_1*CHANNELS/PACKET_CNN];
imap_t input_3[IHEIGHT*IWIDTH_1*CHANNELS/PACKET_CNN];
imap_t input_4[IHEIGHT*IWIDTH_2*CHANNELS/PACKET_CNN];
float input_gru[IHEIGHT*FILTERS];

float outputConv_float[IHEIGHT*FILTERS];
gru_imap_t outputConv_converted[IHEIGHT*FILTERS];
gru_omap_t outputGRU0[IHEIGHT*(GRU_FILTERS*2)];
gru_omap_t outputGRU1[IHEIGHT*(GRU_FILTERS*2)];
float outputGRU1_float[IHEIGHT*(GRU_FILTERS*2)];

float outputTD0[IHEIGHT*FILTERS];
float outputLS[IHEIGHT];
float outputGS[1];

/* output expected validation */
float output_expect_GRU0[IHEIGHT*(GRU_FILTERS*2)];
float output_expect_GRU1[IHEIGHT*(GRU_FILTERS*2)];
float output_expect_LS[IHEIGHT];
float output_expect_GS[1];


weigth_t kernel_0[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET_CNN];
weigth_t kernel_0_scale[CHANNELS/PACKET_CNN];
bias_t bias_0[CHANNELS];

weigth_t kernel_1[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET_CNN];
weigth_t kernel_1_scale[CHANNELS/PACKET_CNN];
bias_t bias_1[CHANNELS];

weigth_t kernel_2[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET_CNN];
weigth_t kernel_2_scale[CHANNELS/PACKET_CNN];
bias_t bias_2[CHANNELS];

weigth_t kernel_3[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET_CNN];
weigth_t kernel_3_scale[CHANNELS/PACKET_CNN];
bias_t bias_3[CHANNELS];

weigth_t kernel_4[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET_CNN];
weigth_t kernel_4_scale[CHANNELS/PACKET_CNN];
bias_t bias_4[CHANNELS];

weigth_t kernel_5[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET_CNN];
weigth_t kernel_5_scale[CHANNELS/PACKET_CNN];
bias_t bias_5[CHANNELS];


gru_weigth_t gru0f_kernel[GRU0_KERNEL_SIZE];
gru_weigth_t gru0f_rkernel[GRU_RKERNEL_SIZE];
gru_weigth_t gru0f_bias[GRU_BIAS_SIZE];
gru_weigth_t gru0f_rbias[GRU_BIAS_SIZE];

gru_weigth_t gru0b_kernel[GRU0_KERNEL_SIZE];
gru_weigth_t gru0b_rkernel[GRU_RKERNEL_SIZE];
gru_weigth_t gru0b_bias[GRU_BIAS_SIZE];
gru_weigth_t gru0b_rbias[GRU_BIAS_SIZE];


gru_weigth_t gru1f_kernel[GRU1_KERNEL_SIZE];
gru_weigth_t gru1f_rkernel[GRU_RKERNEL_SIZE];
gru_weigth_t gru1f_bias[GRU_BIAS_SIZE];
gru_weigth_t gru1f_rbias[GRU_BIAS_SIZE];

gru_weigth_t gru1b_kernel[GRU1_KERNEL_SIZE];
gru_weigth_t gru1b_rkernel[GRU_RKERNEL_SIZE];
gru_weigth_t gru1b_bias[GRU_BIAS_SIZE];
gru_weigth_t gru1b_rbias[GRU_BIAS_SIZE];


float td0_kernel[FILTERS*(FILTERS*2)];
float td0_bias[FILTERS];

float td1_kernel[1*FILTERS];
float td1_bias[1];


void writeBias(bias_t* bias) {
	in_pkt tmp;
	for (int i=0; i<(CHANNELS*BIAS_SIZE/BUS_WIDTH); i++) {
    	tmp.data.range(15,0) = bias[4*i];
    	tmp.data.range(31,16) = bias[4*i+1];
    	tmp.data.range(47,32) = bias[4*i+2];
    	tmp.data.range(63,48) = bias[4*i+3];
    	str_in.write(tmp);
    	printf("bias_tb = 0x%04x 0x%04x 0x%04x 0x%04x\n", (int)bias[i*4], (int)bias[i*4+1], (int)bias[i*4+2], (int)bias[i*4+3]);

    }
}
void writeScale(weigth_t* scale) {
	in_pkt tmp;
    for (int i=0; i<(CHANNELS/PACKET_CNN); i++) {
        tmp.data = scale[i];
        str_in.write(tmp);
    }
}
void writeKernel(weigth_t* kernel) {
	in_pkt tmp;
    for (int i=0; i<(FILTERS*K_SIZE*K_SIZE*CHANNELS/PACKET_CNN); i++) {
        tmp.data = kernel[i];
        if (i==(FILTERS*K_SIZE*K_SIZE*CHANNELS/PACKET_CNN-1)) tmp.last = (ap_int<1>)1;
        else tmp.last = (ap_int<1>)0;
        str_in.write(tmp);
    }
}
void writeInput(imap_t* input) {
	in_pkt tmp;
    for (int i=0, j=0, i_pad=0; i<IHEIGHT*IWIDTH*CHANNELS/PACKET_CNN; i++) {
		ap_uint<6> rangeStart = (j % PACKET_CNN) * 4;
		ap_uint<6> rangeEnd   = rangeStart + 3;
		//printf("i %d, j %d, rs %d, re %d | jP %f\n", i, j, (int)rangeStart, (int)rangeEnd, (float)(j/PACKET_CNN));
		if (i_pad == 0) {
			tmp.data = input[j/PACKET_CNN].range(rangeEnd,rangeStart);
			++j;
		}
		else
			tmp.data = 0x0000000000000000;
		++i_pad;
		if (i_pad == 4) i_pad = 0;
		//printf("writeInput[%d] = 0x%0x 0x%0x\n", i, (int)tmp.data.range(63,32), (int)tmp.data.range(31,0));
        if (i == (IHEIGHT*IWIDTH*CHANNELS/PACKET_CNN-1)) tmp.last = (ap_int<1>)1;
        else tmp.last = (ap_int<1>)0;
        str_in.write(tmp);
    }
}
void writeInputNextLayer() {
	int i = 0;
    in_pkt tmp;
    out_pkt tmpo;
	typedef ap_ufixed<4,0> outputF;
    printf(" --- output of previous layer ---\n");
    while (!str_out.empty()) {
        tmpo = str_out.read();
        int out0 = (int)tmpo.data.range(3,0);	outputF out0F; out0F.range(3,0) = tmpo.data.range(3,0);
        int out1 = (int)tmpo.data.range(7,4);	outputF out1F; out1F.range(3,0) = tmpo.data.range(7,4);
        int out2 = (int)tmpo.data.range(11,8);  outputF out2F; out2F.range(3,0) = tmpo.data.range(11,8);
        int out3 = (int)tmpo.data.range(15,12); outputF out3F; out3F.range(3,0) = tmpo.data.range(15,12);

        int out4 = (int)tmpo.data.range(19,16); outputF out4F; out4F.range(3,0) = tmpo.data.range(19,16);
        int out5 = (int)tmpo.data.range(23,20); outputF out5F; out5F.range(3,0) = tmpo.data.range(23,20);
        int out6 = (int)tmpo.data.range(27,24); outputF out6F; out6F.range(3,0) = tmpo.data.range(27,24);
        int out7 = (int)tmpo.data.range(31,28); outputF out7F; out7F.range(3,0) = tmpo.data.range(31,28);

        int out8 = (int)tmpo.data.range(35,32); outputF out8F; out8F.range(3,0) = tmpo.data.range(35,32);
        int out9 = (int)tmpo.data.range(39,36); outputF out9F; out9F.range(3,0) = tmpo.data.range(39,36);
        int outA = (int)tmpo.data.range(43,40); outputF outAF; outAF.range(3,0) = tmpo.data.range(43,40);
        int outB = (int)tmpo.data.range(47,44); outputF outBF; outBF.range(3,0) = tmpo.data.range(47,44);

        int outC = (int)tmpo.data.range(51,48); outputF outCF; outCF.range(3,0) = tmpo.data.range(51,48);
        int outD = (int)tmpo.data.range(55,52); outputF outDF; outDF.range(3,0) = tmpo.data.range(55,52);
        int outE = (int)tmpo.data.range(59,56); outputF outEF; outEF.range(3,0) = tmpo.data.range(59,56);
        int outF = (int)tmpo.data.range(63,60); outputF outFF; outFF.range(3,0) = tmpo.data.range(63,60);

        printf("%02d - result %2d, %2d, %2d, %2d,   %2d, %2d, %2d, %2d,   %2d, %2d, %2d, %2d,   %2d, %2d, %2d, %2d   "
        		"|   %f, %f, %f, %f,   %f, %f, %f, %f,   %f, %f, %f, %f,   %f, %f, %f, %f\n", i++,
        		out0, out1, out2, out3,
				out4, out5, out6, out7,
				out8, out9, outA, outB,
				outC, outD, outE, outF,
				(float)out0F, (float)out1F, (float)out2F, (float)out3F,
				(float)out4F, (float)out5F, (float)out6F, (float)out7F,
				(float)out8F, (float)out9F, (float)outAF, (float)outBF,
				(float)outCF, (float)outDF, (float)outEF, (float)outFF
		);
    	tmp = tmpo;
    	if (!str_out.empty())
    		tmp.last = (ap_int<1>)0;
    	else
    		tmp.last = (ap_int<1>)1;
    	str_in.write(tmp);
    }
}
void printLastLayerOutput() {
	int i = 0;
    out_pkt tmpo;
	typedef ap_ufixed<4,0> outputF;
    while (!str_out.empty()) {
        tmpo = str_out.read();
        int out0 = (int)tmpo.data.range(3,0);	outputF out0F; out0F.range(3,0) = tmpo.data.range(3,0);
        int out1 = (int)tmpo.data.range(7,4);	outputF out1F; out1F.range(3,0) = tmpo.data.range(7,4);
        int out2 = (int)tmpo.data.range(11,8);  outputF out2F; out2F.range(3,0) = tmpo.data.range(11,8);
        int out3 = (int)tmpo.data.range(15,12); outputF out3F; out3F.range(3,0) = tmpo.data.range(15,12);

        int out4 = (int)tmpo.data.range(19,16); outputF out4F; out4F.range(3,0) = tmpo.data.range(19,16);
        int out5 = (int)tmpo.data.range(23,20); outputF out5F; out5F.range(3,0) = tmpo.data.range(23,20);
        int out6 = (int)tmpo.data.range(27,24); outputF out6F; out6F.range(3,0) = tmpo.data.range(27,24);
        int out7 = (int)tmpo.data.range(31,28); outputF out7F; out7F.range(3,0) = tmpo.data.range(31,28);

        int out8 = (int)tmpo.data.range(35,32); outputF out8F; out8F.range(3,0) = tmpo.data.range(35,32);
        int out9 = (int)tmpo.data.range(39,36); outputF out9F; out9F.range(3,0) = tmpo.data.range(39,36);
        int outA = (int)tmpo.data.range(43,40); outputF outAF; outAF.range(3,0) = tmpo.data.range(43,40);
        int outB = (int)tmpo.data.range(47,44); outputF outBF; outBF.range(3,0) = tmpo.data.range(47,44);

        int outC = (int)tmpo.data.range(51,48); outputF outCF; outCF.range(3,0) = tmpo.data.range(51,48);
        int outD = (int)tmpo.data.range(55,52); outputF outDF; outDF.range(3,0) = tmpo.data.range(55,52);
        int outE = (int)tmpo.data.range(59,56); outputF outEF; outEF.range(3,0) = tmpo.data.range(59,56);
        int outF = (int)tmpo.data.range(63,60); outputF outFF; outFF.range(3,0) = tmpo.data.range(63,60);

        printf("%02d - result %2d, %2d, %2d, %2d,   %2d, %2d, %2d, %2d,   %2d, %2d, %2d, %2d,   %2d, %2d, %2d, %2d   "
        		"|   %f, %f, %f, %f,   %f, %f, %f, %f,   %f, %f, %f, %f,   %f, %f, %f, %f\n", i++,
        		out0, out1, out2, out3,
				out4, out5, out6, out7,
				out8, out9, outA, outB,
				outC, outD, outE, outF,
				(float)out0F, (float)out1F, (float)out2F, (float)out3F,
				(float)out4F, (float)out5F, (float)out6F, (float)out7F,
				(float)out8F, (float)out9F, (float)outAF, (float)outBF,
				(float)outCF, (float)outDF, (float)outEF, (float)outFF
		);
    }
}

void writeWeigthGRU(gru_weigth_t* weigth, size_t size) {
	in_pkt tmp;
    tmp.last = (ap_int<1>)0;
    for (int i=0; i<size; ) {
        tmp.data.range(7,0)   = (int)weigth[i++].range(7,0);
        tmp.data.range(15,8)  = (int)weigth[i++].range(7,0);
        tmp.data.range(23,16) = (int)weigth[i++].range(7,0);
        tmp.data.range(31,24) = (int)weigth[i++].range(7,0);

        tmp.data.range(39,32) = (int)weigth[i++].range(7,0);
        tmp.data.range(47,40) = (int)weigth[i++].range(7,0);
        tmp.data.range(55,48) = (int)weigth[i++].range(7,0);
        tmp.data.range(63,56) = (int)weigth[i++].range(7,0);
        str_in.write(tmp);
    }
}
void conv2gru() {
    /*
     * Currently this function saves the output of last CONV to an array.
     * To send the output to GRU, writeInputNextGRU function should be used.
     */

	int i = 0;
	in_pkt tmp;     // to send to next layer
	out_pkt tmpo;   // to receive from prev layer
	typedef ap_ufixed<4,0> outputF;
    typedef ap_fixed<8,1> inputF;
	printf(" --- conv2gru ---\n");

#ifndef DO_CONV
    printf("INFO: DO_CONV is not defined. Filling output stream for GRU input...\n");
    int cycles = 0;
    while(i<IHEIGHT*FILTERS) {
        outputF out0 = outputConv_float[i++];     tmpo.data.range(3,0)   = out0.range(3,0);
        outputF out1 = outputConv_float[i++];     tmpo.data.range(7,4)   = out1.range(3,0);
        outputF out2 = outputConv_float[i++];     tmpo.data.range(11,8)  = out2.range(3,0);
        outputF out3 = outputConv_float[i++];     tmpo.data.range(15,12) = out3.range(3,0);

        outputF out4 = outputConv_float[i++];     tmpo.data.range(19,16) = out4.range(3,0);
        outputF out5 = outputConv_float[i++];     tmpo.data.range(23,20) = out5.range(3,0);
        outputF out6 = outputConv_float[i++];     tmpo.data.range(27,24) = out6.range(3,0);
        outputF out7 = outputConv_float[i++];     tmpo.data.range(31,28) = out7.range(3,0);

        outputF out8 = outputConv_float[i++];     tmpo.data.range(35,32) = out8.range(3,0);
        outputF out9 = outputConv_float[i++];     tmpo.data.range(39,36) = out9.range(3,0);
        outputF outA = outputConv_float[i++];     tmpo.data.range(43,40) = outA.range(3,0);
        outputF outB = outputConv_float[i++];     tmpo.data.range(47,44) = outB.range(3,0);

        outputF outC = outputConv_float[i++];     tmpo.data.range(51,48) = outC.range(3,0);
        outputF outD = outputConv_float[i++];     tmpo.data.range(55,52) = outD.range(3,0);
        outputF outE = outputConv_float[i++];     tmpo.data.range(59,56) = outE.range(3,0);
        outputF outF = outputConv_float[i++];     tmpo.data.range(63,60) = outF.range(3,0);

        if (i+1 < IHEIGHT*FILTERS)
            tmpo.last = (ap_int<1>)0;
        else
            tmpo.last = (ap_int<1>)1;
        str_out.write(tmpo);

        ++cycles;
    }
    printf("INFO: Stream filled in %d cycles.\n", cycles);
    i = 0;  // reset i to initial value
#endif

	while (!str_out.empty()) {
        /*
        conv -> 4 bits 0 int
        gru  -> 8 bits 1 int
        --------------------
        gru bits convertion conv2gru:
        S -> sign == 0
        C -> conv result
        0 -> padding
        [ SCCC C000 ]
        */
		tmpo = str_out.read();

        int out_offset = 64/2;   // because PACKET_CNN == 16, PACKET_GRU == 8, and that means 2 reads must be made for the convertion
        int out_databits = 64/PACKET_CNN;
        int in_databits = 64/PACKET_GRU;
        for (int section = 0; section < 2; ++section) {
            for (int subOffset = 0; subOffset < 64/PACKET_GRU; ++subOffset) {
                int startPad  = 0;   int endPad  = 2;   // padding
                int startData = 3;   int endData = 6;   // conv result
                int startSign = 7;   int endSign = 7;   // sign bit always positive
                startPad  += in_databits * subOffset;  endPad  += in_databits * subOffset;
                startData += in_databits * subOffset;  endData += in_databits * subOffset;
                startSign += in_databits * subOffset;  endSign += in_databits * subOffset;

                int startOutData = 0;   int endOutData = 3;         // conv result data
                startOutData += out_databits * subOffset + (out_offset * section);
                endOutData   += out_databits * subOffset + (out_offset * section);

                tmp.data.range(endPad, startPad)  = (int)0;
                tmp.data.range(endData,startData) = (int)tmpo.data.range(endOutData,startOutData);
                tmp.data.range(endSign,startSign) = (int)0;

                outputConv_converted[i++].range(7,0) = tmp.data.range(endSign,startPad);
            }
        }
	}
}
void writeInputNextGRU(gru_omap_t* outputGRU, int nCols, int isForward, int isGRU1=0) {
    /* This function send the input in 2 ways:
     * Lets say we have a 2D array 3*3.
     * If isForward == 1, the array is sent LEFT to RIGHT, TOP to BOTTOM.
     * (numbers in the following array are the send order)
     * [
     *  [1 2 3]
     *  [4 5 6]
     *  [7 8 9]
     * ]
     * If isForward == 0, the array is sent LEFT to RIGHT, BOTTOM to TOP.
     * (numbers in the following array are the send order)
     * [
     *  [7 8 9]
     *  [4 5 6]
     *  [1 2 3]
     * ]
     * 
     * NOTE:
     * - THIS FUNCTION WAS ONLY TESTED AND WORKING FOR 64 AND 2 COLUMNS.
     */
	in_pkt tmp;     // to send to next layer

    int idx;
    for (int i = 0; i < IHEIGHT; ++i) {
        // Calculate the LINE based on the direction
        if (isForward) {
            idx = i * nCols;
        } else {
            idx = (IHEIGHT - 1 - i) * nCols;
        }

        // Iterate over COLS
        for (int j = 0; j < nCols; ) {            
            for (int subOffset = 0; subOffset < 64/PACKET_GRU; ++subOffset) {
                int dataBits = 8;
                int startData = 0;                  int endData = dataBits-1;
                startData += dataBits * subOffset;  endData += dataBits * subOffset;
                if (!isGRU1) {
                    tmp.data.range(endData,startData) = outputGRU[idx + j].range(7,0);
                }
                else {
                    if (startData < (8*2)) {
                        /* Why the wierd (8*2) ?
                        * - GRU_1 will read lines with only 2 columns.
                        * - But we are sending 8 values, so the last 6 value must be used as padding with 0.
                        * - This simplifies the GRU_1 "READ_INIT_MAP" loop for now.
                        * - Super unefficient, so a solution to this mess must be found.
                        */
                        tmp.data.range(endData,startData) = outputGRU[idx + j].range(7,0);
                    }
                    else
                        tmp.data.range(endData,startData) = 0;
                }
                ++j;
            }
            if (i<IHEIGHT-1)
                tmp.last = (ap_int<1>)0;
            else
                tmp.last = (ap_int<1>)1;
            str_in.write(tmp);
        }
    }
}
void printGRUoutput(gru_omap_t* output) {
	#define OUT_WIDTH_GRU (GRU_FILTERS*2)
	for (int i=0; i<IHEIGHT; ++i) {
		printf("[%3d] - ", i);
		for (int j=0; j<OUT_WIDTH_GRU; ++j) {
			printf("%10f ", output[(i*OUT_WIDTH_GRU)+j].to_float());
		}
		printf("\n");
    }
}

void gru2td() {
    printf(" --- gru2td ---\n");
    for (int i=0; i<IHEIGHT*(GRU_FILTERS*2); ++i) {
        outputGRU1_float[i] = outputGRU1[i].to_float();
    }
}

// INFO: Currently this method can only compare float arrays!
void compareStats(char* msg, float* actual, float* expected, int size, int breakAfter) {
	printf(msg);
	float* pActual = actual;
	float* pExpected = expected;
	for (int i=0; i<size; ++i) {
		float vActual = *pActual;
		float vExpected = *pExpected;
		float change = vExpected - vActual;
		++pActual;
		++pExpected;

		if (i<breakAfter)
			printf("%12f | %12f | %12f\n", vActual, change, vExpected);
		else {
            printf("     ...     |      ...     |      ...\n");
			break;
        }
	}
}
void compareStats_APF(char* msg, gru_omap_t* actual, float* expected, int size, int breakAfter) {
	printf(msg);
	gru_omap_t* pActual = actual;
	float* pExpected = expected;
	for (int i=0; i<size; ++i) {
		gru_omap_t vActual = *pActual;
		float vExpected = *pExpected;
		float change = vExpected - vActual.to_float();
		++pActual;
		++pExpected;

		if (i<breakAfter)
			printf("%12f | %12f | %12f\n", vActual.to_float(), change, vExpected);
		else {
            printf("     ...     |      ...     |      ...\n");
			break;
        }
	}
}


int main() {
	loadIO(
		input_0,
		outputConv_float,
		output_expect_GRU0,
		output_expect_GRU1,
		output_expect_LS,
		output_expect_GS
	);
    
    loadWeights(
		kernel_0, kernel_0_scale, bias_0,
		kernel_1, kernel_1_scale, bias_1,
		kernel_2, kernel_2_scale, bias_2,
		kernel_3, kernel_3_scale, bias_3,
		kernel_4, kernel_4_scale, bias_4,
		kernel_5, kernel_5_scale, bias_5,
		gru0f_kernel, gru0f_rkernel, gru0f_bias, gru0f_rbias,
		gru0b_kernel, gru0b_rkernel, gru0b_bias, gru0b_rbias,
		gru1f_kernel, gru1f_rkernel, gru1f_bias, gru1f_rbias,
		gru1b_kernel, gru1b_rkernel, gru1b_bias, gru1b_rbias,
		td0_kernel, td0_bias,
		td1_kernel, td1_bias
	);
    loadSigmoidTable();
    loadTanhTable();

    /*
    printf("\ninput_0:\n");
    for (int idx=0; idx<IHEIGHT*IWIDTH*CHANNELS/PACKET_CNN; idx++) {
    	printf("idx=%d | 0x%016llx\n", idx, input_0[idx]);
    }
	*/

#ifndef PRINT_STATS
#ifdef DO_CONV
    printf("CONV_0\n");
    writeBias(bias_0);
    writeScale(kernel_0_scale);
    writeKernel(kernel_0);
    writeInput(input_0);
#if HW_IP
    conv2D(str_in, str_out, 1, IWIDTH);
#endif

    printf("CONV_1\n");
    writeBias(bias_1);
    writeScale(kernel_1_scale);
    writeKernel(kernel_1);
    writeInputNextLayer();
#if HW_IP
    conv2D(str_in, str_out, 2, IWIDTH);
#endif

    printf("CONV_2\n");
    writeBias(bias_2);
    writeScale(kernel_2_scale);
    writeKernel(kernel_2);
    writeInputNextLayer();
#if HW_IP
    conv2D(str_in, str_out, 1, IWIDTH_1);
#endif

    printf("CONV_3\n");
    writeBias(bias_3);
    writeScale(kernel_3_scale);
    writeKernel(kernel_3);
    writeInputNextLayer();
#if HW_IP
    conv2D(str_in, str_out, 2, IWIDTH_1);
#endif

    printf("CONV_4\n");
    writeBias(bias_4);
    writeScale(kernel_4_scale);
    writeKernel(kernel_4);
    writeInputNextLayer();
#if HW_IP
    conv2D(str_in, str_out, 1, IWIDTH_2);
#endif

    printf("CONV_5\n");
    writeBias(bias_5);
    writeScale(kernel_5_scale);
    writeKernel(kernel_5);
    writeInputNextLayer();
#if HW_IP
    conv2D(str_in, str_out, 10, IWIDTH_2);
#endif

#ifdef DEBUG_CONV
    printLastLayerOutput();
#endif

#endif	// !DO_CONV

    conv2gru();
    /*
    for (int i=0; i<431*64; ++i) {
        printf("[%d] = %f", i, outputConv_converted[i].to_float());
        printf("  ---  {%d}-> %f\n", i, outputConv_float[i]);
    }
    */

	printf("###################################### GRU_0 ######################################\n");
	printf("---- 0 FORWARD ----\n");
    writeWeigthGRU(gru0f_kernel, GRU0_KERNEL_SIZE);
    writeWeigthGRU(gru0f_bias, GRU_BIAS_SIZE);
    writeWeigthGRU(gru0f_rkernel, GRU_RKERNEL_SIZE);
    writeWeigthGRU(gru0f_rbias, GRU_BIAS_SIZE);
    writeInputNextGRU(outputConv_converted, FILTERS, GRU_FORWARD);
	gru( // GRU_0_F
		str_in,
		GRU_FORWARD,
		GRU_0__IN_COLS,
        GRU0_KERNEL_SIZE,
		outputGRU0
	);
    printGRUoutput(outputGRU0);

	printf("---- 0 BACKWARD ----\n");
    writeWeigthGRU(gru0b_kernel, GRU0_KERNEL_SIZE);
    writeWeigthGRU(gru0b_bias, GRU_BIAS_SIZE);
    writeWeigthGRU(gru0b_rkernel, GRU_RKERNEL_SIZE);
    writeWeigthGRU(gru0b_rbias, GRU_BIAS_SIZE);
    writeInputNextGRU(outputConv_converted, FILTERS, GRU_BACKWARD);     // send again same output of conv as input
	gru( // GRU_0_B
		str_in,
		GRU_BACKWARD,
		GRU_0__IN_COLS,
        GRU0_KERNEL_SIZE,
		outputGRU0
	);
	printGRUoutput(outputGRU0);

	printf("\n\n\n###################################### GRU_1 ######################################\n");
	printf("---- 1 FORWARD ----\n");
    writeWeigthGRU(gru1f_kernel, GRU1_KERNEL_SIZE);
    writeWeigthGRU(gru1f_bias, GRU_BIAS_SIZE);
    writeWeigthGRU(gru1f_rkernel, GRU_RKERNEL_SIZE);
    writeWeigthGRU(gru1f_rbias, GRU_BIAS_SIZE);
    writeInputNextGRU(outputGRU0, GRU_FILTERS*2, GRU_FORWARD, 1);
	gru( // GRU_1_F
		str_in,
		GRU_FORWARD,
		GRU_FILTERS*2,
		GRU1_KERNEL_SIZE,
		outputGRU1
	);
    
	printf("---- 1 BACKWARD ----\n");
    writeWeigthGRU(gru1b_kernel, GRU1_KERNEL_SIZE);
    writeWeigthGRU(gru1b_bias, GRU_BIAS_SIZE);
    writeWeigthGRU(gru1b_rkernel, GRU_RKERNEL_SIZE);
    writeWeigthGRU(gru1b_rbias, GRU_BIAS_SIZE);
    writeInputNextGRU(outputGRU0, GRU_FILTERS*2, GRU_BACKWARD, 1);
	gru( // GRU_1_B
		str_in,
		GRU_BACKWARD,
		GRU_FILTERS*2,
		GRU1_KERNEL_SIZE,
		outputGRU1
	);
	printGRUoutput(outputGRU1);



	printf("\n\n\n");
	gru2td();

	printf("\n\n\n###################################### TD_0 ######################################\n");
    timedistributed_dense( // TDIST_0 + Dense
    	GRU_FILTERS*2,
        FILTERS, GRU_FILTERS*2,
        TD_0__OUT_COLS,
		outputGRU1_float,
		td0_kernel,
		td0_bias,
        outputTD0
    );

	for (int i=0; i<IHEIGHT; ++i) {
		printf("[%3d] - ", i);
		for (int j=0; j<TD_0__OUT_COLS; ++j) {
			printf("%10f ", outputTD0[(i*TD_0__OUT_COLS)+j]);
		}
		printf("\n");
	}

	printf("\n\n\n###################################### TD_1 ######################################\n");
    timedistributed_dense( // TDIST_1 + Dense
    	TD_0__OUT_COLS,
        1, FILTERS,
        1,
		outputTD0,
		td1_kernel,
		td1_bias,
		outputLS   // LOCAL OUTPUT
    );

	for (int i=0; i<IHEIGHT; ++i) {
		printf("[%3d] - ", i);
		printf("%10f ", outputLS[i]);
		printf("\n");
	}

	printf("\n\n\n###################################### R_MX ######################################\n");
    reducemax_1( // RMAX_1
        outputLS,    // LOCAL OUTPUT -> 431
        outputGS     // GLOBAL OUTPUT -> 1
    );

    printf("GS = %f\n", outputGS[0]);

#ifdef VALIDATE_OUTPUT
    printf("\n\n\n#### #### #### ####\n");
#define BREAK_AFTER 8
    compareStats_APF("Stats GRU_0: (actual | difference | expected)\n", outputGRU0, output_expect_GRU0, IHEIGHT*(GRU_FILTERS*2), BREAK_AFTER);
    compareStats_APF("Stats GRU_1: (actual | difference | expected)\n", outputGRU1, output_expect_GRU1, IHEIGHT*(GRU_FILTERS*2), BREAK_AFTER);
    compareStats("Stats LS: (actual | difference | expected)\n", outputLS, output_expect_LS, IHEIGHT, BREAK_AFTER);
    compareStats("Stats GS: (actual | difference | expected)\n", outputGS, output_expect_GS, 1, BREAK_AFTER);
#endif

#else
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
	float max[27] = {
			0.999719, 0.524543, 0.241185,
			0.456221, 0.415707, 0.322310,
			7.924340, 6.778882, 2.562979,

			0.999959, 0.499182, 0.241185,
			0.557610, 0.551033, 0.422586,
			6.568522, 6.467936, 4.367225,

			10.781307, 0.999979,
			8.283927, 0.999748,
			5.797710, 0.999982,
			0.995374, 0.989570, 0.999959
	};
	float min[27] = {
			-0.987758, -0.434001, -0.148844,
			-0.472353, -0.445030, -0.307011,
			-5.175564, -5.188960, -3.300981,

			-0.999953, -0.572762, -0.148844,
			-0.485191, -0.426595, -0.421792,
			-6.078226, -3.801662, -4.813569,

			-6.392486,  0.000000,
			-6.977444,  0.000000,
			-5.400354, -0.999968,
			-0.994267, -0.997171, -0.999953
	};

#define W_IN 8
#define I_IN 1
	// NOTE: IN, STATE and OUT is advised to be same size
	float max_in = ap_fixed<W_IN,I_IN, AP_RND, AP_SAT>(max[0]).to_float();
	float min_in = ap_fixed<W_IN,I_IN, AP_RND, AP_SAT>(min[0]).to_float();
#define W_K 8
#define I_K 1
	float max_k  = ap_fixed<W_IN,I_IN, AP_RND, AP_SAT>(max[1]).to_float();
	float min_k  = ap_fixed<W_IN,I_IN, AP_RND, AP_SAT>(min[1]).to_float();

#define W_MX_CALC 8
#define I_MX_CALC 1
	float max_mx_calc_0 = ap_fixed<W_MX_CALC,I_MX_CALC, AP_RND, AP_SAT>(max[3]).to_float();
	float min_mx_calc_0 = ap_fixed<W_MX_CALC,I_MX_CALC, AP_RND, AP_SAT>(min[3]).to_float();
	float max_mx_calc_1 = ap_fixed<W_MX_CALC,I_MX_CALC, AP_RND, AP_SAT>(max[4]).to_float();
	float min_mx_calc_1 = ap_fixed<W_MX_CALC,I_MX_CALC, AP_RND, AP_SAT>(min[4]).to_float();
	float max_mx_calc_2 = ap_fixed<W_MX_CALC,I_MX_CALC, AP_RND, AP_SAT>(max[5]).to_float();
	float min_mx_calc_2 = ap_fixed<W_MX_CALC,I_MX_CALC, AP_RND, AP_SAT>(min[5]).to_float();

#define W_B 8
#define I_B 1
	float max_b = ap_fixed<W_B,I_B, AP_RND, AP_SAT>(max[2]).to_float();
	float min_b = ap_fixed<W_B,I_B, AP_RND, AP_SAT>(min[2]).to_float();
#define W_MXB 8
#define I_MXB 4
	float max_mxb_0 = ap_fixed<W_MXB,I_MXB, AP_RND, AP_SAT>(max[6]).to_float();
	float min_mxb_0 = ap_fixed<W_MXB,I_MXB, AP_RND, AP_SAT>(min[6]).to_float();
	float max_mxb_1 = ap_fixed<W_MXB,I_MXB, AP_RND, AP_SAT>(max[7]).to_float();
	float min_mxb_1 = ap_fixed<W_MXB,I_MXB, AP_RND, AP_SAT>(min[7]).to_float();
	float max_mxb_2 = ap_fixed<W_MXB,I_MXB, AP_RND, AP_SAT>(max[8]).to_float();
	float min_mxb_2 = ap_fixed<W_MXB,I_MXB, AP_RND, AP_SAT>(min[8]).to_float();
	///////////////////////////////////////////////////////////////////////

#define W_STATE	8
#define I_STATE	1
	// NOTE: IN, STATE and OUT is advised to be same size
	float max_state = ap_fixed<W_STATE,I_STATE, AP_RND, AP_SAT>(max[9]).to_float();
	float min_state = ap_fixed<W_STATE,I_STATE, AP_RND, AP_SAT>(min[9]).to_float();
#define W_RK 8
#define I_RK 1
	float max_rk    = ap_fixed<W_RK,I_RK, AP_RND, AP_SAT>(max[10]).to_float();
	float min_rk    = ap_fixed<W_RK,I_RK, AP_RND, AP_SAT>(min[10]).to_float();

#define W_MI_CALC 8
#define I_MI_CALC 1
	float max_mi_calc_0 = ap_fixed<W_MI_CALC,I_MI_CALC, AP_RND, AP_SAT>(max[12]).to_float();
	float min_mi_calc_0 = ap_fixed<W_MI_CALC,I_MI_CALC, AP_RND, AP_SAT>(min[12]).to_float();
	float max_mi_calc_1 = ap_fixed<W_MI_CALC,I_MI_CALC, AP_RND, AP_SAT>(max[13]).to_float();
	float min_mi_calc_1 = ap_fixed<W_MI_CALC,I_MI_CALC, AP_RND, AP_SAT>(min[13]).to_float();
	float max_mi_calc_2 = ap_fixed<W_MI_CALC,I_MI_CALC, AP_RND, AP_SAT>(max[14]).to_float();
	float min_mi_calc_2 = ap_fixed<W_MI_CALC,I_MI_CALC, AP_RND, AP_SAT>(min[14]).to_float();

#define W_RB 8
#define I_RB 1
	float max_rb = ap_fixed<W_RB,I_RB, AP_RND, AP_SAT>(max[11]).to_float();
	float min_rb = ap_fixed<W_RB,I_RB, AP_RND, AP_SAT>(min[11]).to_float();
#define W_MIB 8
#define I_MIB 4
	float max_mib_0 = ap_fixed<W_MIB,I_MIB, AP_RND, AP_SAT>(max[15]).to_float();
	float min_mib_0 = ap_fixed<W_MIB,I_MIB, AP_RND, AP_SAT>(min[15]).to_float();
	float max_mib_1 = ap_fixed<W_MIB,I_MIB, AP_RND, AP_SAT>(max[16]).to_float();
	float min_mib_1 = ap_fixed<W_MIB,I_MIB, AP_RND, AP_SAT>(min[16]).to_float();
	float max_mib_2 = ap_fixed<W_MIB,I_MIB, AP_RND, AP_SAT>(max[17]).to_float();
	float min_mib_2 = ap_fixed<W_MIB,I_MIB, AP_RND, AP_SAT>(min[17]).to_float();
	///////////////////////////////////////////////////////////////////////

#define W_ZSIG_CALC 16
#define I_ZSIG_CALC 8
	// W 8 and I 4 -> saturated at 7, value expected 10
	float max_zsig_calc = ap_fixed<W_ZSIG_CALC,I_ZSIG_CALC, AP_RND, AP_SAT>(max[18]).to_float();
	float min_zsig_calc = ap_fixed<W_ZSIG_CALC,I_ZSIG_CALC, AP_RND, AP_SAT>(min[18]).to_float();
#define W_Z 8
#define I_Z 0
	float max_z         = ap_ufixed<W_Z,I_Z, AP_RND, AP_SAT>(max[19]).to_float();
	float min_z         = ap_ufixed<W_Z,I_Z, AP_RND, AP_SAT>(min[19]).to_float();

#define W_RSIG_CALC 8
#define I_RSIG_CALC 4
	float max_rsig_calc = ap_fixed<W_RSIG_CALC,I_RSIG_CALC, AP_RND, AP_SAT>(max[20]).to_float();
	float min_rsig_calc = ap_fixed<W_RSIG_CALC,I_RSIG_CALC, AP_RND, AP_SAT>(min[20]).to_float();
#define W_R 8
#define I_R 0
	float max_r         = ap_ufixed<W_R,I_R, AP_RND, AP_SAT>(max[21]).to_float();
	float min_r         = ap_ufixed<W_R,I_R, AP_RND, AP_SAT>(min[21]).to_float();

#define W_HHTANH_CALC 8
#define I_HHTANH_CALC 4
	float max_hhtanh_calc = ap_fixed<W_HHTANH_CALC,I_HHTANH_CALC, AP_RND, AP_SAT>(max[22]).to_float();
	float min_hhtanh_calc = ap_fixed<W_HHTANH_CALC,I_HHTANH_CALC, AP_RND, AP_SAT>(min[22]).to_float();
#define W_HH 8 
#define I_HH 1
	float max_hh          = ap_fixed<W_HH,I_HH, AP_RND, AP_SAT>(max[23]).to_float();
	float min_hh          = ap_fixed<W_HH,I_HH, AP_RND, AP_SAT>(min[23]).to_float();

#define W_ZSTATE_CALC 8
#define I_ZSTATE_CALC 1
	float max_zstate_calc = ap_fixed<W_ZSTATE_CALC,I_ZSTATE_CALC, AP_RND, AP_SAT>(max[24]).to_float();
	float min_zstate_calc = ap_fixed<W_ZSTATE_CALC,I_ZSTATE_CALC, AP_RND, AP_SAT>(min[24]).to_float();
#define W_ZHH_CALC 8
#define I_ZHH_CALC 1
	float max_zhh_calc    = ap_fixed<W_ZHH_CALC,I_ZHH_CALC, AP_RND, AP_SAT>(max[25]).to_float();
	float min_zhh_calc    = ap_fixed<W_ZHH_CALC,I_ZHH_CALC, AP_RND, AP_SAT>(min[25]).to_float();
#define W_OUT 8
#define I_OUT 1
	// NOTE: IN, STATE and OUT is advised to be same size
	float max_out         = ap_fixed<W_OUT,I_OUT, AP_RND, AP_SAT>(max[26]).to_float();
	float min_out         = ap_fixed<W_OUT,I_OUT, AP_RND, AP_SAT>(min[26]).to_float();

	printf( " ---- MAX ----\n"
			"%f %f %f\n"
			"%f %f %f\n"
			"%f %f %f\n\n"
			"%f %f %f\n"
			"%f %f %f\n"
			"%f %f %f\n\n"
			"%f %f\n"
			"%f %f\n"
			"%f %f\n"
			"%f %f %f\n\n\n",
			max_in, max_k, max_b,
			max_mx_calc_0, max_mx_calc_1, max_mx_calc_2,
			max_mxb_0, max_mxb_1, max_mxb_2,

			max_state, max_rk, max_rb,
			max_mi_calc_0, max_mi_calc_1, max_mi_calc_2,
			max_mib_0, max_mib_1, max_mib_2,

			max_zsig_calc, max_z,
			max_rsig_calc, max_r,
			max_hhtanh_calc, max_hh,
			max_zstate_calc, max_zhh_calc, max_out
	);
	printf( " ---- MIN ----\n"
			"%f %f %f\n"
			"%f %f %f\n"
			"%f %f %f\n\n"
			"%f %f %f\n"
			"%f %f %f\n"
			"%f %f %f\n\n"
			"%f %f\n"
			"%f %f\n"
			"%f %f\n"
			"%f %f %f\n\n\n",
			min_in, min_k, min_b,
			min_mx_calc_0, min_mx_calc_1, min_mx_calc_2,
			min_mxb_0, min_mxb_1, min_mxb_2,

			min_state, min_rk, min_rb,
			min_mi_calc_0, min_mi_calc_1, min_mi_calc_2,
			min_mib_0, min_mib_1, min_mib_2,

			min_zsig_calc, min_z,
			min_rsig_calc, min_r,
			min_hhtanh_calc, min_hh,
			min_zstate_calc, min_zhh_calc, min_out
	);
#endif // !PRINT_STATS

	/*
		for (float value=-1; value<=1; value += 0.01) {
			float truncated = ap_fixed<4,1, AP_RND, AP_SAT>(value).to_float();
			printf("VAL = %f | TC = %f\n", value, truncated);
		}
	*/

    int err_cnt = 0;
    return 0;
}
