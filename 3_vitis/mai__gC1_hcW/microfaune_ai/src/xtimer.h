#ifndef XTIMER_H
#define XTIMER_H

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


void getLastTimes(struct Times* times);

#endif
