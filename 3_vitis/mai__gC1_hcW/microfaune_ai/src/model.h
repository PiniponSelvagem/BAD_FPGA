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


#include "xtime_l.h"

#define DO_TIMES
#define TIMER_COUNTS_PER_SECOND		COUNTS_PER_SECOND
struct Times {
	XTime conv0;
	XTime conv1;
	XTime conv2;
	XTime conv3;
	XTime conv4;
	XTime conv5;
	XTime conv2gru;
	XTime bgru_0;
	XTime bgru_1;
	XTime gru2td;
	XTime timedist_0;
	XTime timedist_1;
	XTime reducemax_1;
};

Times xtimes = {};
void getLastTimes(Times* times) {
	times->conv0 = xtimes.conv0;
	times->conv1 = xtimes.conv1;
	times->conv2 = xtimes.conv2;
	times->conv3 = xtimes.conv3;
	times->conv4 = xtimes.conv4;
	times->conv5 = xtimes.conv5;
	times->conv2gru = xtimes.conv2gru;
	times->bgru_0 = xtimes.bgru_0;
	times->bgru_1 = xtimes.bgru_1;
	times->gru2td = xtimes.gru2td;
	times->timedist_0 = xtimes.timedist_0;
	times->timedist_1 = xtimes.timedist_1;
	times->reducemax_1 = xtimes.reducemax_1;
}

#ifdef DO_TIMES
#define START_TIMER		XTime_GetTime(&tStart);
#define STOP_TIMER(name) \
		XTime_GetTime(&tEnd); \
		xtimes.name = (tEnd - tStart);
#else
#define START_TIMER		 ;
#define STOP_TIMER(name) ;
#endif



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
	#ifdef DO_TIMES
	XTime tStart, tEnd;
	#endif // DO_TIMES

		START_TIMER
	conv3d_0(input);
		STOP_TIMER(conv0)
		START_TIMER
	conv3d_1();
		STOP_TIMER(conv1)
		START_TIMER
	conv3d_2();
		STOP_TIMER(conv2)
		START_TIMER
	conv3d_3();
		STOP_TIMER(conv3)
		START_TIMER
	conv3d_4();
		STOP_TIMER(conv4)
		START_TIMER
	unsigned char* outConv = conv3d_5();
		STOP_TIMER(conv5)

		START_TIMER
	conv2gru(outConv, CONV_OUTPUT_SIZE_5_IN_BYTES, inGru);
	Xil_DCacheInvalidateRange((INTPTR)(inGru), GRU_INPUT_SIZE_0_IN_BYTES);
		STOP_TIMER(conv2gru)

		START_TIMER
	bgru_0(inGru /*input_gru*/, outputArray0);
		STOP_TIMER(bgru_0)
		START_TIMER
	bgru_1(outputArray0, outputArray1);
		STOP_TIMER(bgru_1)

		START_TIMER
	gru2td(outputArray1, RNN_OUT_SIZE, outputGRU);
		STOP_TIMER(gru2td)

		START_TIMER
	timedist_0(outputGRU, outputTD0);
		STOP_TIMER(timedist_0)
		START_TIMER
	timedist_1(outputTD0, outputLS);
		STOP_TIMER(timedist_1)
		START_TIMER
	reducemax_1(outputLS, outputGS);
		STOP_TIMER(reducemax_1)

	return outputGS[0];
}


#endif // MODEL_H
