#include <stdio.h>
#include <stdlib.h>
#include "xaxidma.h"
#include "xparameters.h"
#include "xgru.h"

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
    #include "input_0_gru_50124.h"
#elif SELECTED_INPUT == INPUT_BIRD_1
    //
#elif SELECTED_INPUT == INPUT_BIRD_2
    //
#elif SELECTED_INPUT == INPUT_BIRD_3
    //
#elif SELECTED_INPUT == INPUT_NO_BIRD_0
    //
#elif SELECTED_INPUT == INPUT_NO_BIRD_1
    //
#elif SELECTED_INPUT == INPUT_NO_BIRD_2
    //
#elif SELECTED_INPUT == INPUT_NO_BIRD_3
    //
#else
    #error "Invalid INPUT definition"
#endif
/****************************************************/

/* AXI Lite */
#define DEV_ID_LITE		XPAR_GRU_0_DEVICE_ID
XGru AxiLite;

#define DMA_DEV_ID_STREAM	XPAR_AXIDMA_0_DEVICE_ID
XAxiDma AxiDma;
/* ******** */


#define ALIGN_TO_BYTE(value) (((value) + 7) & ~7)

/* AXI Stream */
#define LINES					431
#define GRU_CELLS				1
#define INPUT_ARRAY_ELEMS		(LINES*64)
#define INPUT_ARRAY_SIZE		INPUT_ARRAY_ELEMS	// "unsigned char" and "8bits" values

// GRU_0
#define INPUT_SIZE_0_IN_BYTES	INPUT_ARRAY_SIZE
#define OUTPUT_SIZE_0_IN_BYTES	ALIGN_TO_BYTE((LINES*GRU_CELLS))		// the smallest transfer is 8 bytes, align at byte

#define RX_ADDR	0x200000
#define TX_ADDR 0x400000
/* ********** */


void HW_gru(unsigned int *input, unsigned int input_size, unsigned int *output, unsigned int output_size, int layerIndex) {
	printf("START: HW_gru\r\n");

	printf("Sending layerIndex = %d (AXI Lite)\r\n", layerIndex);
	/* send config */
	XGru_Set_layerIndex(&AxiLite, layerIndex);

	printf("Gru_start\r\n");
	XGru_Start(&AxiLite);

	int status;
	//volatile unsigned int *rxBufferPtr = (unsigned int*)RX_ADDR;
	//volatile unsigned int *txBufferPtr = (unsigned int*)TX_ADDR;

	//txBufferPtr = input;

	//printf("InvalidateRange of input with size %d\r\n", input_size);
	//Xil_DCacheInvalidateRange((INTPTR)(input), (unsigned)(input_size));
	/*
	printf("InvalidateRange of output with size %d\r\n", output_size);
	Xil_DCacheInvalidateRange((INTPTR)(output), (unsigned)(output_size));
	*/

	/* receive OUTPUT */
	// configure receive before core starts working
	//rxBufferPtr = (unsigned int*)output;
	printf("SimpleTransfer: Rx\r\n");
	status = XAxiDma_SimpleTransfer(&AxiDma,(UINTPTR)output, output_size, XAXIDMA_DEVICE_TO_DMA);
	if (status != XST_SUCCESS) {
		printf("Error: %d | Failed sending: output\n", status);
		return;
	}

	/* send INPUT */
	//txBufferPtr = (unsigned int*)input;
	printf("SimpleTransfer: Tx\r\n");
	status = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)input, input_size, XAXIDMA_DMA_TO_DEVICE);
	if (status != XST_SUCCESS) {
		printf("Error: %d | Failed sending: input\n", status);
		return;
	}

	printf("Wait for Tx...\r\n");
	while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE)) { /* Wait for Tx*/ }
	printf("Tx done.\r\n");

	printf("Wait for Rx...\r\n");
	while (XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) { /* Wait for Rx*/	}
	printf("Rx done.\r\n");

/*
	/* return the received OUTPUT *
	for (unsigned int i = 0; i < (output_size/8)*2; ++i) {	// div 8 -> 32bits/4bits = 8 values   |   mult 2 -> 32 bits int and we using 64 bits
		//printf("output[%d] = 0x%08x\r\n", i, rxBufferPtr[i]);
		output[i] = rxBufferPtr[i];
	}
*/
	/*
	//Xil_DCacheInvalidateRange((INTPTR)(input), (unsigned)(input_size));
	Xil_DCacheInvalidateRange((INTPTR)(txBufferPtr), (unsigned)(input_size));
	Xil_DCacheInvalidateRange((INTPTR)(rxBufferPtr), (unsigned)(output_size));
	*/

	printf("END: HW_gru\r\n");
}


// Initialize AXI lite device
int init_XAxi_lite_SimplePollMode(u16 deviceID) {
	XGru_Config* cfgPtr;
	int status;

	cfgPtr = XGru_LookupConfig(deviceID);
	if (!cfgPtr) {
		printf("No config found for device %d\r\n", deviceID);
		return XST_FAILURE;
	}

	status = XGru_CfgInitialize(&AxiLite, cfgPtr);
	if (status != XST_SUCCESS) {
		printf("Initialization failed with status %d\r\n", status);
		return XST_FAILURE;
	}

	//printf("DisableAutoRestart\n");
	// Disable autorestart & interrupts
	//XAxil_conv2d0_DisableAutoRestart(&AxiConv2D0);
	//XAxil_conv2d0_InterruptDisable(&AxiConv2D0, 0xFFFF);
	//XAxil_conv2d0_InterruptGlobalDisable(&AxiConv2D0);

	return XST_SUCCESS;
}

// Initialize DMA stream device
int init_XAxiDma_stream_SimplePollMode(u16 deviceID) {
	XAxiDma_Config* cfgPtr;
	int status;

	cfgPtr = XAxiDma_LookupConfig(deviceID);
	if (!cfgPtr) {
		printf("No config found for device %d\r\n", deviceID);
		return XST_FAILURE;
	}

	status = XAxiDma_CfgInitialize(&AxiDma, cfgPtr);
	if (status != XST_SUCCESS) {
		printf("Initialization failed with status %d\r\n", status);
		return XST_FAILURE;
	}

	if (XAxiDma_HasSg(&AxiDma)) {
		/**
		 * 1st test - it entered here.
		 * In vivado disable in the DMA: "Enable Scatter Gather Engine"
		 */
		printf("Device configured as SG mode \r\n");
		return XST_FAILURE;
	}

	/* Disable interrupts, we use polling mode	 */
	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

	return XST_SUCCESS;
}


void flipArrayLines(unsigned char* array, size_t size, int nCols) {
	/* Lets say we have a 2D array 3*3.
	 * The array will be flip by lines and will keep the columns in the original order.
	 * (numbers in the following array are the send order)
	 * Original:
	 * [
	 *  [1 2 3]
	 *  [4 5 6]
	 *  [7 8 9]
	 * ]
	 * Result:
	 * [
	 *  [7 8 9]
	 *  [4 5 6]
	 *  [1 2 3]
	 * ]
	 */
    size_t start = 0;
    size_t end = size - nCols;

    while (start < end) {
        // Swap elements at start and end indices for each row
        for (int i = 0; i < nCols; i++) {
            unsigned char temp = array[start + i];
            array[start + i] = array[end + i];
            array[end + i] = temp;
        }

        // Move indices towards the center for the next row
        start += nCols;
        end -= nCols;
    }
}

/**
 * direction: 1 forward, 0 backward
 */
void saveOutputTo(unsigned char* input, size_t insize, unsigned char* output, int direction) {
    size_t inputIndex = 0;
    size_t outputIndex;

    if (direction == 1) {
        outputIndex = 0;
    } else {
        outputIndex = 2*insize - GRU_CELLS;
    }

    while (inputIndex < insize) {
    	for (int i = 0; i < GRU_CELLS; ++i) {
			if (direction == 1) {
				output[outputIndex] = input[inputIndex];
			} else {
				output[outputIndex] = input[inputIndex];
			}
			++inputIndex;
			++outputIndex;
		}

    	if (direction == 1) {
    		outputIndex += GRU_CELLS;
		} else {
			outputIndex -= (2 * GRU_CELLS) + GRU_CELLS;
		}
    }
}

#define OUT_SIZE	((LINES*GRU_CELLS)*2)
unsigned char outputArray[OUT_SIZE] = {0};

int main() {
	printf("\r\n#### "__DATE__" "__TIME__" ####\r\n");
	int status;

	// AXI Lite
	status = init_XAxi_lite_SimplePollMode(DEV_ID_LITE);
	if (status != XST_SUCCESS) {
		printf("init_XAxi_lite_SimplePollMode: Failed\r\n");
		return XST_FAILURE;
	}

	// AXI Stream
	status = init_XAxiDma_stream_SimplePollMode(DMA_DEV_ID_STREAM);
	if (status != XST_SUCCESS) {
		printf("init_XAxiDma_stream_SimplePollMode: Failed\r\n");
		return XST_FAILURE;
	}

	// FORWARD
	HW_gru(
			(unsigned int*)input_gru, INPUT_SIZE_0_IN_BYTES,
			(unsigned int*)RX_ADDR, OUTPUT_SIZE_0_IN_BYTES,
			0
	);
	Xil_DCacheInvalidateRange((INTPTR)(RX_ADDR), OUTPUT_SIZE_0_IN_BYTES);
	saveOutputTo((unsigned char*)RX_ADDR, LINES, outputArray, 1);


	// BACKWARD
	flipArrayLines(input_gru, INPUT_SIZE_0_IN_BYTES, 64);
	Xil_DCacheInvalidateRange((INTPTR)(input_gru), INPUT_SIZE_0_IN_BYTES);

	HW_gru(
			(unsigned int*)input_gru, INPUT_SIZE_0_IN_BYTES,
			(unsigned int*)RX_ADDR, OUTPUT_SIZE_0_IN_BYTES,
			1
	);

	Xil_DCacheInvalidateRange((INTPTR)(RX_ADDR), OUTPUT_SIZE_0_IN_BYTES);
	saveOutputTo((unsigned char*)RX_ADDR, LINES, outputArray, 0);

	printf("outArray\r\n");
	for (unsigned int i=0, l=0; i<OUT_SIZE; ++l) {
		printf("[%3d] - 0x%02x 0x%02x\r\n", l, outputArray[i++], outputArray[i++]);
	}


	return 0;
}
