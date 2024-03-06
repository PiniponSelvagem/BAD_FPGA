#include <stdio.h>
#include <stdlib.h>

#include "hardlayer/bgru.h"

#include "softlayer/timedist.h"
#include "softlayer/reducemax.h"


float outputGRU[(IHEIGHT*GRU_CELLS)*2];
float outputTD0[IHEIGHT*FILTERS];
float outputLS[IHEIGHT];
float outputGS[1];

#include <unistd.h>		// TODO: remove, being used for sleep on a print


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
#define SELECTED_INPUT INPUT_BIRD_0

#if SELECTED_INPUT == INPUT_BIRD_0
    #include "inputs/input_0_gru_50124.h"
#elif SELECTED_INPUT == INPUT_BIRD_1
	#include "inputs/input_1_gru_52046.h"
#elif SELECTED_INPUT == INPUT_BIRD_2
    #include "inputs/input_2_gru_16835.h"
#elif SELECTED_INPUT == INPUT_BIRD_3
    #include "inputs/input_3_gru_80705.h"
#elif SELECTED_INPUT == INPUT_NO_BIRD_0
    #include "inputs/input_4_gru_50678.h"
#elif SELECTED_INPUT == INPUT_NO_BIRD_1
    #include "inputs/input_5_gru_51034.h"
#elif SELECTED_INPUT == INPUT_NO_BIRD_2
    #include "inputs/input_6_gru_1931.h"
#elif SELECTED_INPUT == INPUT_NO_BIRD_3
    #include "inputs/input_7_gru_79266.h"
#else
    #error "Invalid INPUT definition"
#endif
/****************************************************/



#define RNN_OUT_SIZE	((IHEIGHT*GRU_CELLS)*2)
unsigned char outputArray0[RNN_OUT_SIZE] = {0};
unsigned char outputArray1[RNN_OUT_SIZE] = {0};

int main() {
	printf("\r\n#### " __DATE__ " " __TIME__ " ####\r\n");

	int status = init_bgru();
	if (status != 0) {
		printf("Bidirectional GRU failed initializing with code: %d\r\n", status);
		return status;
	}


	bgru_0(input_gru, outputArray0);
	bgru_1(outputArray0, outputArray1);

	printf("          GRU_0   |   GRU_1\r\n");
	for (unsigned int a=0, b=0, l=0; a<RNN_OUT_SIZE; ++l) {
		printf("[%3d] - 0x%02x 0x%02x | 0x%02x 0x%02x\r\n", l, outputArray0[a++], outputArray0[a++], outputArray1[b++], outputArray1[b++]);
	}


	gru2td(outputArray1, RNN_OUT_SIZE, outputGRU);

	printf("#### TD_0 ####\n");
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

	printf("#### TD_1 ####\n");
	timedist_1(outputTD0, outputLS);
    /*
	for (int i=0; i<IHEIGHT; ++i) {
		printf("[%3d] - %10f\n", i, outputLS[i]);
		if (i>95)
			sleep(2);
	}
	*/

	printf("#### R_MX ####\n");
    reducemax_1( // RMAX_1
        outputLS,    // LOCAL OUTPUT -> 431
        outputGS     // GLOBAL OUTPUT -> 1
    );

    printf("Global Score = %f\n", outputGS[0]);

	return 0;
}
