#ifndef MODEL_H
#define MODEL_H

#include "global_settings.h"

#include "hardlayer/conv3d.h"
#include "hardlayer/bgru.h"

#include "softlayer/timedist.h"
#include "softlayer/reducemax.h"

#include "xtime_l.h"


/* AUX arrays */
unsigned char inGru[GRU_INPUT_SIZE_0_IN_BYTES];

#define RNN_OUT_SIZE	((IHEIGHT*GRU_CELLS)*2)
unsigned char outputArray0[RNN_OUT_SIZE] = {0};
unsigned char outputArray1[RNN_OUT_SIZE] = {0};

float outputGRU[(IHEIGHT*GRU_CELLS)*2];
float outputTD0[IHEIGHT*FILTERS];
float outputLS[IHEIGHT];
float outputGS[1];
/* ********** */


int init_model() {
	int status = init_conv3d();
	if (status != 0) {
		printf("Convolution 3D failed initializing with code: %d\r\n", status);
		return status;
	}
	status = init_bgru();
	if (status != 0) {
		printf("Bidirectional GRU failed initializing with code: %d\r\n", status);
		return status;
	}
	return 0;
}

float modelPredict(const unsigned char* input) {
	XTime tStart, tEnd;

	XTime_GetTime(&tStart);
	conv3d_0(input);
	conv3d_1();
	conv3d_2();
	conv3d_3();
	conv3d_4();
	unsigned char* outConv = conv3d_5();
	XTime_GetTime(&tEnd);


	printf("\nTime: Conv3D\n");
	printf("> Clock cycles: %llu\n", 1*(tEnd - tStart));
	printf("> %.3f us\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000));
	printf("> %.3f ms\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000));
	printf("> %.3f s\n", 1.0 * (tEnd - tStart) / COUNTS_PER_SECOND);


	XTime_GetTime(&tStart);
	conv2gru(outConv, CONV_OUTPUT_SIZE_5_IN_BYTES, inGru);
	Xil_DCacheInvalidateRange((INTPTR)(inGru), GRU_INPUT_SIZE_0_IN_BYTES);
	XTime_GetTime(&tEnd);

	printf("\nTime: conv2gru\n");
	printf("> Clock cycles: %llu\n", 1*(tEnd - tStart));
	printf("> %.3f us\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000));
	printf("> %.3f ms\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000));
	printf("> %.3f s\n", 1.0 * (tEnd - tStart) / COUNTS_PER_SECOND);

	/*
	unsigned int *out1 = (unsigned int*)outConv;
#define EXIT1		(CONV_OUTPUT_SIZE_5_IN_BYTES/8)*2	// div 8 -> 32bits/4bits = 8 values   |   mult 2 -> 32 bits int and we using 64 bits
	for (unsigned int i=0; i<EXIT1; ++i) {
		printf("outConv[%d] = 0x%08x\r\n", i, out1[i]);
	}
	*/

	/*
	// validate gru input
	for (unsigned int i=0; i<27584; ++i) {
		unsigned char expected = input_gru[i];
		unsigned char actual = inGru[i];
		if (actual != expected)
			printf("i=%d | expected=%0x02 - actual=0x%02x\r\n", i, expected, actual);
	}
	*/

	XTime_GetTime(&tStart);
	bgru_0(inGru /*input_gru*/, outputArray0);
	bgru_1(outputArray0, outputArray1);
	XTime_GetTime(&tEnd);

	printf("\nTime: BGRU\n");
	printf("> Clock cycles: %llu\n", 1*(tEnd - tStart));
	printf("> %.3f us\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000));
	printf("> %.3f ms\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000));
	printf("> %.3f s\n", 1.0 * (tEnd - tStart) / COUNTS_PER_SECOND);

	/*
	printf("          GRU_0   |   GRU_1\r\n");
	for (unsigned int a=0, b=0, l=0; a<RNN_OUT_SIZE; ++l) {
		printf("[%3d] - 0x%02x 0x%02x | 0x%02x 0x%02x\r\n", l, outputArray0[a++], outputArray0[a++], outputArray1[b++], outputArray1[b++]);
	}
	*/


	gru2td(outputArray1, RNN_OUT_SIZE, outputGRU);

	timedist_0(outputGRU, outputTD0);
	/*
	for (int i=0; i<IHEIGHT; ++i) {
		printf("[%3d] - ", i);
		for (int j=0; j<FILTERS; ++j) {
			printf("%10f ", outputTD0[(i*FILTERS)+j]);
		}
		printf("\n");
	}
	*/

	timedist_1(outputTD0, outputLS);
	/*
	for (int i=0; i<IHEIGHT; ++i) {
		printf("[%3d] - %10f\n", i, outputLS[i]);
		if (i>95)
			sleep(2);
	}
	*/

	reducemax_1(outputLS, outputGS);

	return outputGS[0];
}


#endif // MODEL_H
