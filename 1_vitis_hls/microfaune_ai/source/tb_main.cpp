#define HW_IP 1
#include "hls_stream.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>

#include "types.h"
#include "size_conv3D.h"
#include "size_bgru.h"

#include "load_weights.h"


typedef ap_axis<64, 0, 0, 0> in_pkt;
typedef ap_axis<64, 0, 0, 0> out_pkt;

hls::stream<in_pkt> str_in;
hls::stream<out_pkt> str_out;


void conv2D(hls::stream<in_pkt> &strm_in, hls::stream<out_pkt> &strm_out, int pool, int maxWidth);
void gru(
	int isForward,
	int kernelCols,
	gru_t* input,
	gru_t* kernel,    gru_t* bias,
	gru_t* recKernel, gru_t* recBias,
	gru_t* output
);

imap_t input_0[IHEIGHT*IWIDTH/PACKET];
imap_t input_1[IHEIGHT*IWIDTH*CHANNELS/PACKET];
imap_t input_2[IHEIGHT*IWIDTH_1*CHANNELS/PACKET];
imap_t input_3[IHEIGHT*IWIDTH_1*CHANNELS/PACKET];
imap_t input_4[IHEIGHT*IWIDTH_2*CHANNELS/PACKET];
gru_t input_gru[IHEIGHT*FILTERS];

gru_t outputConv[IHEIGHT*FILTERS];
gru_t outputGRU0[IHEIGHT*(FILTERS*2)];
gru_t outputGRU1[IHEIGHT*(FILTERS*2)];


weigth_t kernel_0[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_0_scale[CHANNELS/PACKET];
bias_t bias_0[CHANNELS];

weigth_t kernel_1[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_1_scale[CHANNELS/PACKET];
bias_t bias_1[CHANNELS];

weigth_t kernel_2[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_2_scale[CHANNELS/PACKET];
bias_t bias_2[CHANNELS];

weigth_t kernel_3[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_3_scale[CHANNELS/PACKET];
bias_t bias_3[CHANNELS];

weigth_t kernel_4[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_4_scale[CHANNELS/PACKET];
bias_t bias_4[CHANNELS];

weigth_t kernel_5[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_5_scale[CHANNELS/PACKET];
bias_t bias_5[CHANNELS];



gru_t gru0f_kernel[GRU_IN_COLS*GRU_SPLIT_SIZE*GRU_FILTERS];
gru_t gru0f_rkernel[GRU_FILTERS*GRU_SPLIT_SIZE*GRU_FILTERS];
gru_t gru0f_bias[GRU_BIAS_SIZE];
gru_t gru0f_rbias[GRU_BIAS_SIZE];

gru_t gru0b_kernel[GRU_IN_COLS*GRU_SPLIT_SIZE*GRU_FILTERS];
gru_t gru0b_rkernel[GRU_FILTERS*GRU_SPLIT_SIZE*GRU_FILTERS];
gru_t gru0b_bias[GRU_BIAS_SIZE];
gru_t gru0b_rbias[GRU_BIAS_SIZE];


gru_t gru1f_kernel[(GRU_FILTERS*2)*GRU_SPLIT_SIZE*GRU_FILTERS];
gru_t gru1f_rkernel[GRU_FILTERS*GRU_SPLIT_SIZE*GRU_FILTERS];
gru_t gru1f_bias[GRU_BIAS_SIZE];
gru_t gru1f_rbias[GRU_BIAS_SIZE];

gru_t gru1b_kernel[(GRU_FILTERS*2)*GRU_SPLIT_SIZE*GRU_FILTERS];
gru_t gru1b_rkernel[GRU_FILTERS*GRU_SPLIT_SIZE*GRU_FILTERS];
gru_t gru1b_bias[GRU_BIAS_SIZE];
gru_t gru1b_rbias[GRU_BIAS_SIZE];




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
    for (int i=0; i<(CHANNELS/PACKET); i++) {
        tmp.data = scale[i];
        str_in.write(tmp);
    }
}
void writeKernel(weigth_t* kernel) {
	in_pkt tmp;
    for (int i=0; i<(FILTERS*K_SIZE*K_SIZE*CHANNELS/PACKET); i++) {
        tmp.data = kernel[i];
        if (i==(FILTERS*K_SIZE*K_SIZE*CHANNELS/PACKET-1)) tmp.last = (ap_int<1>)1;
        else tmp.last = (ap_int<1>)0;
        str_in.write(tmp);
    }
}
void writeInput(imap_t* input) {
	in_pkt tmp;
    for (int i=0, j=0, i_pad=0; i<IHEIGHT*IWIDTH*CHANNELS/PACKET; i++) {
		ap_uint<6> rangeStart = (j % PACKET) * 4;
		ap_uint<6> rangeEnd   = rangeStart + 3;
		//printf("i %d, j %d, rs %d, re %d | jP %f\n", i, j, (int)rangeStart, (int)rangeEnd, (float)(j/PACKET));
		if (i_pad == 0) {
			tmp.data = input[j/PACKET].range(rangeEnd,rangeStart);
			++j;
		}
		else
			tmp.data = 0x0000000000000000;
		++i_pad;
		if (i_pad == 4) i_pad = 0;
		//printf("writeInput[%d] = 0x%0x 0x%0x\n", i, (int)tmp.data.range(63,32), (int)tmp.data.range(31,0));
        if (i == (IHEIGHT*IWIDTH*CHANNELS/PACKET-1)) tmp.last = (ap_int<1>)1;
        else tmp.last = (ap_int<1>)0;
        str_in.write(tmp);
    }
}
void writeInput3(imap_t* input) {
	in_pkt tmp;
    for (int i=0; i<IHEIGHT*20*CHANNELS/PACKET; i++) {
		//printf("i %d, j %d, rs %d, re %d | jP %f\n", i, j, (int)rangeStart, (int)rangeEnd, (float)(j/PACKET));
		tmp.data = input[i].range(63,0);
        if (i == (IHEIGHT*20*CHANNELS/PACKET-1)) tmp.last = (ap_int<1>)1;
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
void conv2gru() {
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
        		"|   %f, %f, %f, %f,   %f, %f, %f, %f,   %f, %f, %f, %f,   %f, %f, %f, %f\n", i,
        		out0, out1, out2, out3,
				out4, out5, out6, out7,
				out8, out9, outA, outB,
				outC, outD, outE, outF,
				(float)out0F, (float)out1F, (float)out2F, (float)out3F,
				(float)out4F, (float)out5F, (float)out6F, (float)out7F,
				(float)out8F, (float)out9F, (float)outAF, (float)outBF,
				(float)outCF, (float)outDF, (float)outEF, (float)outFF
		);

        outputConv[i++] = (float)out0F;
        outputConv[i++] = (float)out1F;
        outputConv[i++] = (float)out2F;
        outputConv[i++] = (float)out3F;

        outputConv[i++] = (float)out4F;
        outputConv[i++] = (float)out5F;
        outputConv[i++] = (float)out6F;
        outputConv[i++] = (float)out7F;

        outputConv[i++] = (float)out8F;
        outputConv[i++] = (float)out9F;
        outputConv[i++] = (float)outAF;
        outputConv[i++] = (float)outBF;

        outputConv[i++] = (float)outCF;
        outputConv[i++] = (float)outDF;
        outputConv[i++] = (float)outEF;
        outputConv[i++] = (float)outFF;
    }
}
void printGRUoutput(gru_t* output) {
	#define OUT_WIDTH_GRU (FILTERS*2)
	for (int i=0; i<IHEIGHT; ++i) {
		printf("[%3d] - ", i);
		for (int j=0; j<OUT_WIDTH_GRU; ++j) {
			printf("%10f ", output[(i*OUT_WIDTH_GRU)+j]);
		}
		printf("\n");
    }
}

//#define DEBUG_CONV
//#define DO_CONV
int main() {
    loadWeights(
		input_0, input_1, input_2, input_3, input_4,
		(float*)outputConv,
		kernel_0, kernel_0_scale, bias_0,
		kernel_1, kernel_1_scale, bias_1,
		kernel_2, kernel_2_scale, bias_2,
		kernel_3, kernel_3_scale, bias_3,
		kernel_4, kernel_4_scale, bias_4,
		kernel_5, kernel_5_scale, bias_5,
		gru0f_kernel, gru0f_rkernel, gru0f_bias, gru0f_rbias,
		gru0b_kernel, gru0b_rkernel, gru0b_bias, gru0b_rbias,
		gru1f_kernel, gru1f_rkernel, gru1f_bias, gru1f_rbias,
		gru1b_kernel, gru1b_rkernel, gru1b_bias, gru1b_rbias
	);

    /*
    printf("\ninput_0:\n");
    for (int idx=0; idx<IHEIGHT*IWIDTH*CHANNELS/PACKET; idx++) {
    	printf("idx=%d | 0x%016llx\n", idx, input_0[idx]);
    }
	*/

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

#ifdef DEBUG_MODEL
    printLastLayerOutput();
#endif

#ifndef DEBUG_MODEL
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
    //writeInput3(input_3);
#if HW_IP
    conv2D(str_in, str_out, 2, IWIDTH_1);
#endif

    //printLastLayerOutput();	// REMOVE AFTER DEBUG CONV_3 WITH writeInput3


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
#endif

    // Temporary method to convert from hardware to software data.
    printf("---- conv2gru ----\n");
    conv2gru();
#else
    printf("---- outputConv from TF ----\n");
	for (int i=0; i<IHEIGHT; ++i) {
		printf("[%3d] - ", i);
		for (int j=0; j<(FILTERS); ++j) {
			printf("%10f ", outputConv[(i*FILTERS)+j]);
		}
		printf("\n");
    }
#endif	// DO_CONV

	printf("###################################### GRU_0 ######################################\n");
	printf("---- 0 FORWARD ----\n");
	gru( // GRU_0_F
		GRU_FORWARD,
		GRU_0__IN_COLS,
		outputConv,
		gru0f_kernel,	gru0f_bias,
		gru0f_rkernel,	gru0f_rbias,
		outputGRU0
	);

	printf("---- 0 BACKWARD ----\n");
	gru( // GRU_0_B
		GRU_BACKWARD,
		GRU_0__IN_COLS,
		outputConv,
		gru0b_kernel,	gru0b_bias,
		gru0b_rkernel, 	gru0b_rbias,
		outputGRU0
	);

	printGRUoutput(outputGRU0);


	printf("\n\n\n###################################### GRU_1 ######################################\n");
	printf("---- 1 FORWARD ----\n");
	gru( // GRU_1_F
		GRU_FORWARD,
		GRU_FILTERS*2,
		outputGRU0,
		gru1f_kernel,	gru1f_bias,
		gru1f_rkernel,	gru1f_rbias,
		outputGRU1
	);

	printf("---- 1 BACKWARD ----\n");
	gru( // GRU_1_B
		GRU_BACKWARD,
		GRU_FILTERS*2,
		outputGRU0,
		gru1b_kernel,  	gru1b_bias,
		gru1b_rkernel, 	gru1b_rbias,
		outputGRU1
	);

	printGRUoutput(outputGRU1);

    int err_cnt = 0;
    return 0;
}
