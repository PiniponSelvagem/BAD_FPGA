#ifndef BGRU_H
#define BGRU_H

#include "xaxidma.h"
#include "xparameters.h"
#include "xgru.h"

#include "../global_settings.h"

#include "../utils/arrays.h"
#include "../utils/convertion.h"

//#define DEBUG_GRU


/* AXI Stream */
// GRU_0
#define GRU_INPUT_ARRAY_SIZE_0				(IHEIGHT*FILTERS)
#define GRU_INPUT_SIZE_0_IN_BYTES			GRU_INPUT_ARRAY_SIZE_0
#define GRU_OUTPUT_SIZE_0_IN_BYTES			ALIGN_TO_BYTE((IHEIGHT*GRU_CELLS))	// per direction
#define GRU_OUTPUT_SIZE_0_IN_BYTES_BGRU		(GRU_OUTPUT_SIZE_0_IN_BYTES*2)			// both directions, end of BGRU_0

// GRU_1
#define GRU_INPUT_ARRAY_SIZE_1				((IHEIGHT*GRU_CELLS)*2)
#define GRU_INPUT_SIZE_1_IN_BYTES			ALIGN_TO_BYTE(GRU_INPUT_ARRAY_SIZE_1)
#define GRU_OUTPUT_SIZE_1_IN_BYTES			ALIGN_TO_BYTE((IHEIGHT*GRU_CELLS))	// per direction
#define GRU_OUTPUT_SIZE_1_IN_BYTES_BGRU		(GRU_OUTPUT_SIZE_1_IN_BYTES*2)			// both directions, end of BGRU_1


#define BUFFER_GRU_ADDR		0x600000
/* ********** */



/* AXI Lite */
#define DEV_ID_GRU		XPAR_GRU_0_DEVICE_ID
XGru AxiLiteGru;

#define DMA_DEV_ID_STREAM_GRU	XPAR_AXIDMA_1_DEVICE_ID
XAxiDma AxiDmaGru;
/* ******** */



// Initialize AXI lite device
int init_XAxiLiteGru_SimplePollMode(u16 deviceID) {
	XGru_Config* cfgPtr;
	int status;

	cfgPtr = XGru_LookupConfig(deviceID);
	if (!cfgPtr) {
		printf("No config found for device %d\r\n", deviceID);
		return XST_FAILURE;
	}

	status = XGru_CfgInitialize(&AxiLiteGru, cfgPtr);
	if (status != XST_SUCCESS) {
		printf("Initialization failed with status %d\r\n", status);
		return XST_FAILURE;
	}

	return XST_SUCCESS;
}

// Initialize DMA stream device
int init_XAxiDmaGru_stream_SimplePollMode(u16 deviceID) {
	XAxiDma_Config* cfgPtr;
	int status;

	cfgPtr = XAxiDma_LookupConfig(deviceID);
	if (!cfgPtr) {
		printf("No config found for device %d\r\n", deviceID);
		return XST_FAILURE;
	}

	status = XAxiDma_CfgInitialize(&AxiDmaGru, cfgPtr);
	if (status != XST_SUCCESS) {
		printf("Initialization failed with status %d\r\n", status);
		return XST_FAILURE;
	}

	if (XAxiDma_HasSg(&AxiDmaGru)) {
		/**
		 * 1st test - it entered here.
		 * In vivado disable in the DMA: "Enable Scatter Gather Engine"
		 */
		printf("Device configured as SG mode \r\n");
		return XST_FAILURE;
	}

	/* Disable interrupts, we use polling mode	 */
	XAxiDma_IntrDisable(&AxiDmaGru, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&AxiDmaGru, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

	return XST_SUCCESS;
}


int init_bgru() {
	int status;

	// AXI Lite
	status = init_XAxiLiteGru_SimplePollMode(DEV_ID_GRU);
	if (status != XST_SUCCESS) {
		printf("init_XAxiLiteGru_SimplePollMode: Failed\r\n");
		return XST_FAILURE;
	}

	// AXI Stream
	status = init_XAxiDmaGru_stream_SimplePollMode(DMA_DEV_ID_STREAM_GRU);
	if (status != XST_SUCCESS) {
		printf("init_XAxiDmaGru_stream_SimplePollMode: Failed\r\n");
		return XST_FAILURE;
	}

	return 0;
}


void HW_bgru(unsigned int *input, unsigned int input_size, unsigned int *output, unsigned int output_size, int layerIndex) {
#ifdef DEBUG_GRU
	printf("START: HW_bgru\r\n");
	printf("Sending layerIndex = %d (AXI Lite)\r\n", layerIndex);
#endif
	/* send layer index */
	XGru_Set_layerIndex(&AxiLiteGru, layerIndex);

#ifdef DEBUG_GRU
	printf("Gru_start\r\n");
#endif
	XGru_Start(&AxiLiteGru);

#ifdef DEBUG_GRU
	printf("SimpleTransfer: Rx (%d bytes)\r\n", output_size);
#endif
	/* receive OUTPUT */
	int status;
	status = XAxiDma_SimpleTransfer(&AxiDmaGru,(UINTPTR)output, output_size, XAXIDMA_DEVICE_TO_DMA);
	if (status != XST_SUCCESS) {
		printf("Error: %d | Failed sending: output\n", status);
		return;
	}

#ifdef DEBUG_GRU
	printf("SimpleTransfer: Tx (%d bytes)\r\n", input_size);
#endif
	/* send INPUT */
	status = XAxiDma_SimpleTransfer(&AxiDmaGru, (UINTPTR)input, input_size, XAXIDMA_DMA_TO_DEVICE);
	if (status != XST_SUCCESS) {
		printf("Error: %d | Failed sending: input\n", status);
		return;
	}

#ifdef DEBUG_GRU
	printf("Wait for Tx...\r\n");
#endif
	while (XAxiDma_Busy(&AxiDmaGru, XAXIDMA_DMA_TO_DEVICE)) { /* Wait for Tx*/ }
#ifdef DEBUG_GRU
	printf("Tx done.\r\n");

	printf("Wait for Rx...\r\n");
#endif
	while (XAxiDma_Busy(&AxiDmaGru, XAXIDMA_DEVICE_TO_DMA)) { /* Wait for Rx*/	}
#ifdef DEBUG_GRU
	printf("Rx done.\r\n");

	printf("END: HW_bgru\r\n");
#endif
}

void bgru(
		unsigned char* input, int inSizeBytes,
		int forwardIndex, int backwardIndex,
		int bufferSizeBytes,
		int flipSizeElems, int flipNCols,
		unsigned char* output, int outSizeBytes
	) {
	// FORWARD
	HW_bgru(
			(unsigned int*)input, inSizeBytes,
			(unsigned int*)BUFFER_GRU_ADDR, bufferSizeBytes,
			forwardIndex
	);
	Xil_DCacheInvalidateRange((INTPTR)(BUFFER_GRU_ADDR), bufferSizeBytes);
	saveGruOutputTo((unsigned char*)BUFFER_GRU_ADDR, IHEIGHT, output, 1, GRU_CELLS);

	// prepare input for backward
	flipArrayLines(input, flipSizeElems, flipNCols);
	Xil_DCacheInvalidateRange((INTPTR)(input), flipSizeElems);

	// BACKWARD
	HW_bgru(
			(unsigned int*)input, inSizeBytes,
			(unsigned int*)BUFFER_GRU_ADDR, bufferSizeBytes,
			backwardIndex
	);
	Xil_DCacheInvalidateRange((INTPTR)(BUFFER_GRU_ADDR), bufferSizeBytes);
	saveGruOutputTo((unsigned char*)BUFFER_GRU_ADDR, IHEIGHT, output, 0, GRU_CELLS);
	Xil_DCacheInvalidateRange((INTPTR)(output), outSizeBytes);
}

void bgru_0(unsigned char* input, unsigned char* output) {
	bgru(
			input, GRU_INPUT_SIZE_0_IN_BYTES,
			0, 1,
			GRU_OUTPUT_SIZE_0_IN_BYTES,
			GRU_INPUT_ARRAY_SIZE_0, FILTERS,
			output, GRU_OUTPUT_SIZE_0_IN_BYTES_BGRU
	);
}
void bgru_1(unsigned char* input, unsigned char* output) {

	bgru(
			input, GRU_INPUT_SIZE_1_IN_BYTES,
			2, 3,
			GRU_OUTPUT_SIZE_1_IN_BYTES,
			GRU_INPUT_ARRAY_SIZE_1, GRU_CELLS*2,
			output, GRU_OUTPUT_SIZE_1_IN_BYTES_BGRU
	);
}


#endif // BGRU_H
