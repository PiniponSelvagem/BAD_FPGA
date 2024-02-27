#include <stdio.h>
#include <stdlib.h>
#include "xaxidma.h"
#include "xparameters.h"
#include "xconv2d.h"

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
    #include "input_0_50124.h"
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
#define DEV_ID_LITE		XPAR_CONV2D_0_DEVICE_ID
XConv2d AxiLite;

#define DMA_DEV_ID_STREAM	XPAR_AXIDMA_0_DEVICE_ID
XAxiDma AxiDma;
/* ******** */

/* AXI Stream */
#define INPUT_ARRAY_ELEMS		(2*40*64)
#define INPUT_ARRAY_SIZE		(INPUT_ARRAY_ELEMS/2)	// divided by 2 -> ARRAY is "unsigned char" and "8/2 = 4bits"

// Conv2D_0
#define INPUT_SIZE_0_IN_BYTES	INPUT_ARRAY_SIZE
#define OUTPUT_SIZE_0_IN_BYTES	(INPUT_SIZE_0_IN_BYTES)

// Conv2D_1
#define INPUT_SIZE_1_IN_BYTES	OUTPUT_SIZE_0_IN_BYTES
#define OUTPUT_SIZE_1_IN_BYTES	(INPUT_SIZE_1_IN_BYTES/2)

// Conv2D_2
#define INPUT_SIZE_2_IN_BYTES	OUTPUT_SIZE_1_IN_BYTES
#define OUTPUT_SIZE_2_IN_BYTES	INPUT_SIZE_2_IN_BYTES

// Conv2D_3
#define INPUT_SIZE_3_IN_BYTES	OUTPUT_SIZE_2_IN_BYTES
#define OUTPUT_SIZE_3_IN_BYTES	(INPUT_SIZE_3_IN_BYTES/2)

// Conv2D_4
#define INPUT_SIZE_4_IN_BYTES	OUTPUT_SIZE_3_IN_BYTES
#define OUTPUT_SIZE_4_IN_BYTES	INPUT_SIZE_4_IN_BYTES

// Conv2D_5
#define INPUT_SIZE_5_IN_BYTES	OUTPUT_SIZE_4_IN_BYTES
#define OUTPUT_SIZE_5_IN_BYTES	(INPUT_SIZE_5_IN_BYTES/10)

#define RX_ADDR	0x200000
#define TX_ADDR 0x400000
/* ********** */


void HW_conv2d(unsigned int *input, unsigned int input_size, unsigned int *output, unsigned int output_size, int layerIndex) {
	printf("START: HW_conv2d\r\n");

	printf("Sending layerIndex = %d (AXI Lite)\r\n", layerIndex);
	/* send config */
	XConv2d_Set_layerIndex(&AxiLite, layerIndex);

	printf("Conv2d_start\r\n");
	XConv2d_Start(&AxiLite);

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

	printf("END: HW_conv2d\r\n");
}


// Initialize AXI lite device
int init_XAxi_lite_SimplePollMode(u16 deviceID) {
	XConv2d_Config* cfgPtr;
	int status;

	cfgPtr = XConv2d_LookupConfig(deviceID);
	if (!cfgPtr) {
		printf("No config found for device %d\r\n", deviceID);
		return XST_FAILURE;
	}

	status = XConv2d_CfgInitialize(&AxiLite, cfgPtr);
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


unsigned char output0[INPUT_ARRAY_SIZE];
unsigned char output1[INPUT_ARRAY_SIZE];

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



	HW_conv2d(
			(unsigned int*)input, INPUT_SIZE_0_IN_BYTES,
			(unsigned int*)RX_ADDR, OUTPUT_SIZE_0_IN_BYTES,
			0
	);

	/*
	unsigned int *out0 = (unsigned int*)output0;
#define EXIT0		(OUTPUT_SIZE_0_IN_BYTES/8)*2	// div 8 -> 32bits/4bits = 8 values   |   mult 2 -> 32 bits int and we using 64 bits
	for (unsigned int i=0; i<EXIT0; ++i) {
		if (i<8 || i>(EXIT0-1)-8)
			printf("out0[%d] = 0x%08x\r\n", i, out0[i]);
	}
	*/


	HW_conv2d(
			(unsigned int*)RX_ADDR, INPUT_SIZE_1_IN_BYTES,
			(unsigned int*)TX_ADDR, OUTPUT_SIZE_1_IN_BYTES,
			1
	);

	HW_conv2d(
			(unsigned int*)TX_ADDR, INPUT_SIZE_2_IN_BYTES,
			(unsigned int*)RX_ADDR, OUTPUT_SIZE_2_IN_BYTES,
			2
	);

	HW_conv2d(
			(unsigned int*)RX_ADDR, INPUT_SIZE_3_IN_BYTES,
			(unsigned int*)TX_ADDR, OUTPUT_SIZE_3_IN_BYTES,
			3
	);

	HW_conv2d(
			(unsigned int*)TX_ADDR, INPUT_SIZE_4_IN_BYTES,
			(unsigned int*)RX_ADDR, OUTPUT_SIZE_4_IN_BYTES,
			4
	);

	HW_conv2d(
			(unsigned int*)RX_ADDR, INPUT_SIZE_5_IN_BYTES,
			(unsigned int*)TX_ADDR, OUTPUT_SIZE_5_IN_BYTES,
			5
	);

	Xil_DCacheInvalidateRange((INTPTR)(TX_ADDR), OUTPUT_SIZE_5_IN_BYTES);
	unsigned int *out1 = (unsigned int*)TX_ADDR;
#define EXIT1		(OUTPUT_SIZE_5_IN_BYTES/8)*2	// div 8 -> 32bits/4bits = 8 values   |   mult 2 -> 32 bits int and we using 64 bits
	for (unsigned int i=0; i<EXIT1; ++i) {
		printf("out1[%d] = 0x%08x\r\n", i, out1[i]);
	}


/*
	config = 2;
	HW_lite_stream(input, output, config);

	config = 1;
	HW_lite_stream(input, output, config);
*/
	/*
	for (int i = 0; i < ARRAY_SIZE; ++i) {
		printf("input[%d] = %d\r\n", i, input[i]);
	}
	*/
/*
	for (int i = 0; i < INPUT_ARRAY_SIZE; ++i) {
		printf("output[%d] = 0x%02x\r\n", i, output[i]);
	}
*/

	return 0;
}
