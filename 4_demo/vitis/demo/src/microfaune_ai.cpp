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
	printf("%s", percentage);
#endif
}





unsigned char* inputNoPad = (unsigned char*)0x1000000;
unsigned char* inputFile  = (unsigned char*)0x100F000;	// D:\BAD_FPGA\3_vitis\mai__gC1_hcW\microfaune_ai\input_pack\input_400.bin
#define INPUT_BYTES_NOPAD	((431*40)/2)
#define INPUT_BYTES			(INPUT_BYTES_NOPAD*64)

#define PADDING 64
void addPadding(unsigned char* in, unsigned char* out) {
	char* pin  = (char*)in;
	char* pout = (char*)out;
    for (int i=0; i<INPUT_BYTES_NOPAD; ++i) {
    	char valH = (*pin & 0xF0) >> 4;
    	char valL = *pin & 0x0F;

    	*pout = valL;
    	++pout;
    	for (int j=0; j<((PADDING/2)-1); ++j) {
    		*pout = 0;
    		++pout;
    	}

    	*pout = valH;
    	++pout;
    	for (int j=0; j<((PADDING/2)-1); ++j) {
    		*pout = 0;
    		++pout;
    	}

    	++pin;
    }
}

#include "xuartps.h"  // Include the UART peripheral header file
int main() {
	//printf("\r\n#### "__DATE__" "__TIME__" ####\r\n");

	int status = init_model();
	if (status != 0) {
		//printf("Model failed initializing with code: %d\r\n", status);
		return status;
	}

	XUartPs_Config *ConfigPtr;
	XUartPs uart;
	ConfigPtr = XUartPs_LookupConfig(XPAR_PSU_UART_1_DEVICE_ID);
	XUartPs_CfgInitialize(&uart, ConfigPtr, ConfigPtr->BaseAddress);

	while (1) {
		int bytesRead = 0;
#define BUFF_SIZE	32
		char buffer[BUFF_SIZE];
		struct Times times = {};

		while (bytesRead < INPUT_BYTES_NOPAD) {		// TODO: handle incomplete RX by ignoring them
			// Wait until data is available
			while (!(XUartPs_IsReceiveData(uart.Config.BaseAddress)))
				;

			// Read data from UART
			bytesRead += XUartPs_Recv(&uart, (u8 *)(inputNoPad + bytesRead), INPUT_BYTES_NOPAD - bytesRead);
		}

		addPadding(inputNoPad, inputFile);
		Xil_DCacheInvalidateRange((INTPTR)(inputFile), INPUT_BYTES);

		float percentage = modelPredict(inputFile);
		getLastTimes(&times);
		floatToString(percentage, buffer, BUFF_SIZE, 6);
		printResult(0, buffer, &times);

		XUartPs_Send(&uart, (u8*)buffer, 8);
	}

	return 0;
}
