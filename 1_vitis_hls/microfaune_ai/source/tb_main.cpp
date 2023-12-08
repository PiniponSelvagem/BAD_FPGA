#define HW_IP 1
#include "hls_stream.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>

#include "types.h"
#include "size_conv3D.h"

#include "load_weights.h"

typedef ap_axis<64, 0, 0, 0> in_pkt;
typedef ap_axis<64, 0, 0, 0> out_pkt;

hls::stream<in_pkt> str_in;
hls::stream<out_pkt> str_out;


void conv2D(hls::stream<in_pkt> &strm_in, hls::stream<out_pkt> &strm_out, int pool, int maxWidth);

imap_t input_1[IHEIGHT*IWIDTH/PACKET];


weigth_t kernel_0[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_0_scale[CHANNELS/PACKET];
weigth_t bias_0[CHANNELS/PACKET];

weigth_t kernel_1[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_1_scale[CHANNELS/PACKET];
weigth_t bias_1[CHANNELS/PACKET];

weigth_t kernel_2[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_2_scale[CHANNELS/PACKET];
weigth_t bias_2[CHANNELS/PACKET];

weigth_t kernel_3[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_3_scale[CHANNELS/PACKET];
weigth_t bias_3[CHANNELS/PACKET];

weigth_t kernel_4[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_4_scale[CHANNELS/PACKET];
weigth_t bias_4[CHANNELS/PACKET];

weigth_t kernel_5[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_5_scale[CHANNELS/PACKET];
weigth_t bias_5[CHANNELS/PACKET];

void writeBias(weigth_t* bias) {
	in_pkt tmp;
    for (int i=0; i<(CHANNELS/PACKET); i++) {
        tmp.data = bias[i];
        str_in.write(tmp);
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
    for (int i=0, j=0; i<IHEIGHT*IWIDTH*CHANNELS/PACKET; i++) {
		ap_uint<6> rangeStart = (j % PACKET) * 4;
		ap_uint<6> rangeEnd   = rangeStart + 3;
		//printf("i %d, j %d, rs %d, re %d\n", i, j, (int)rangeStart, (int)rangeEnd);
		tmp.data = input[j/PACKET].range(rangeEnd,rangeStart);
		++j;
        if (i == (IHEIGHT*IWIDTH*CHANNELS/PACKET-1)) tmp.last = (ap_int<1>)1;
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
    	tmp.last = (ap_int<1>)0;
    	str_in.write(tmp);
    }
	tmp.last = (ap_int<1>)1;
	str_in.write(tmp);
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

int main() {
    loadWeights(
		input_1,	// input_2,
		kernel_0, kernel_0_scale, bias_0,
		kernel_1, kernel_1_scale, bias_1,
		kernel_2, kernel_2_scale, bias_2,
		kernel_3, kernel_3_scale, bias_3,
		kernel_4, kernel_4_scale, bias_4,
		kernel_5, kernel_5_scale, bias_5
	);


    printf("CONV_0\n");
    writeBias(bias_0);
    writeScale(kernel_0_scale);
    writeKernel(kernel_0);
    writeInput(input_1);
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
#if HW_IP
    conv2D(str_in, str_out, 2, IWIDTH_1);
#endif

    printf("CONV_4\n");
    writeBias(bias_2);
    writeScale(kernel_2_scale);
    writeKernel(kernel_2);
    writeInputNextLayer();
#if HW_IP
    conv2D(str_in, str_out, 1, IWIDTH_2);
#endif

    printf("CONV_5\n");
    writeBias(bias_3);
    writeScale(kernel_3_scale);
    writeKernel(kernel_3);
    writeInputNextLayer();
#if HW_IP
    conv2D(str_in, str_out, 10, IWIDTH_2);
#endif

    printLastLayerOutput();
#endif


    int err_cnt = 0;
    return 0;
}
