#include <stdio.h>
#include <stdlib.h>

#include "xtime_l.h"

#include "model.h"

#include "floatToString.h"

/******************* SELECT INPUT *******************/
#define INPUT_BIRD_0 0      // expected 1 cell: 0.9706467986106873   | expected 64 cells: 0.9919541478157043
#define INPUT_BIRD_1 1      // expected 1 cell: 0.5680659413337708   | expected 64 cells: 0.9428370594978333
#define INPUT_BIRD_2 2      // expected 1 cell: 0.9324995875358582   | expected 64 cells: 0.9812303781509399
#define INPUT_BIRD_3 3      // expected 1 cell: 0.9636806845664978   | expected 64 cells: 0.9797500371932983

#define INPUT_NO_BIRD_0 4   // expected 1 cell: 0.041791435331106186 | expected 64 cells: 0.0710141733288765
#define INPUT_NO_BIRD_1 5   // expected 1 cell: 0.40294381976127625  | expected 64 cells: 0.12189839780330658
#define INPUT_NO_BIRD_2 6   // expected 1 cell: 0.061949472874403    | expected 64 cells: 0.041453707963228226
#define INPUT_NO_BIRD_3 7   // expected 1 cell: 0.12713807821273804  | expected 64 cells: 0.0648108646273613

// SELECT INPUT
//#define SELECTED_INPUT 	INPUT_BIRD_0
#define SELECTED_INPUT	-1

#if SELECTED_INPUT == INPUT_BIRD_0
    #include "inputs/input_0_50124.h"
	//#include "inputs/gru/input_0_gru_50124.h"
#elif SELECTED_INPUT == INPUT_BIRD_1
	#include "inputs/input_1_52046.h"
#elif SELECTED_INPUT == INPUT_BIRD_2
    #include "inputs/input_2_16835.h"
#elif SELECTED_INPUT == INPUT_BIRD_3
    #include "inputs/input_3_80705.h"
#elif SELECTED_INPUT == INPUT_NO_BIRD_0
    #include "inputs/input_4_80678.h"
#elif SELECTED_INPUT == INPUT_NO_BIRD_1
    #include "inputs/input_5_51034.h"
#elif SELECTED_INPUT == INPUT_NO_BIRD_2
    #include "inputs/input_6_1931.h"
#elif SELECTED_INPUT == INPUT_NO_BIRD_3
    #include "inputs/input_7_79266.h"
#else
    //#error "Invalid INPUT definition"
#endif
/****************************************************/


#ifdef DO_TIMES
#define SHOW_TIMES_STR  "%lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu"
#define SHOW_TIMES_VALS \
	times->conv0, times->conv1, times->conv2, times->conv3, times->conv4, times->conv5, \
	times->conv2gru, \
	times->bgru_0, times->bgru_1, \
	times->gru2td, \
	times->timedist_0, times->timedist_1, \
	times->reducemax_1
#endif
void printResult(int idx, char* percentage, struct Times* times) {
#ifdef DO_TIMES
	printf("input[%03d] = %s | " SHOW_TIMES_STR "\n", idx, percentage, SHOW_TIMES_VALS);
#else
	printf("input[%03d] = %s\n", idx, buffer);
#endif
}


#define INPUT_SIZE	400
const unsigned char* inputFile = (unsigned char*)0x1000000;	// D:\BAD_FPGA\3_vitis\mai__gC1_hcW\microfaune_ai\input_pack\input_400.bin


int main() {
	printf("\r\n#### "__DATE__" "__TIME__" ####\r\n");

	int status = init_model();
	if (status != 0) {
		printf("Model failed initializing with code: %d\r\n", status);
		return status;
	}

#define BUFF_SIZE	20
	char buffer[BUFF_SIZE];
	struct Times times = {};
#ifdef DO_TIMES
	printf("COUNTS_PER_SECOND = %d | Times are in clock cycles -> TIME/COUNTS_PER_SECOUNT -> seconds\n", COUNTS_PER_SECOND);
#endif // DO_TIMES
	const unsigned char* input = inputFile;
	for (int i=0; i<INPUT_SIZE; ++i) {
		float percentage = modelPredict(input);
		getLastTimes(&times);
		floatToString(percentage, buffer, BUFF_SIZE, 6);
		printResult(i, buffer, &times);
		input += (431*(64/2)*40);
	}

/*
	//XTime tStart, tEnd;
	//XTime_GetTime(&tStart);
	float percentage = modelPredict(input);
	//XTime_GetTime(&tEnd);

	printf("Global Score = %f\n", percentage);
*/
	/*
	printf("\nTime:\n");
	printf("> Clock cycles: %llu\n", 1*(tEnd - tStart));
	printf("> %.3f us\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000));
	printf("> %.3f ms\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000));
	printf("> %.3f s\n", 1.0 * (tEnd - tStart) / COUNTS_PER_SECOND);
	*/
	return 0;
}
