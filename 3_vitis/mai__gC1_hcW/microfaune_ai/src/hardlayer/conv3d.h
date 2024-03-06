#ifndef CONV3D_H
#define CONV3D_H

#include "xaxidma.h"
#include "xparameters.h"
#include "xconv2d.h"

#include "../global_settings.h"

//#define DEBUG_CONV


/* AXI Stream */
#define CONV_INPUT_ARRAY_ELEMS		(431*40*64)
#define CONV_INPUT_ARRAY_SIZE		(CONV_INPUT_ARRAY_ELEMS/2)	// divided by 2 -> ARRAY is "unsigned char" and "8/2 = 4bits"

// Conv3D_0
#define CONV_INPUT_SIZE_0_IN_BYTES	CONV_INPUT_ARRAY_SIZE
#define CONV_OUTPUT_SIZE_0_IN_BYTES	(CONV_INPUT_SIZE_0_IN_BYTES)

// Conv3D_1
#define CONV_INPUT_SIZE_1_IN_BYTES	CONV_OUTPUT_SIZE_0_IN_BYTES
#define CONV_OUTPUT_SIZE_1_IN_BYTES	(CONV_INPUT_SIZE_1_IN_BYTES/2)

// Conv3D_2
#define CONV_INPUT_SIZE_2_IN_BYTES	CONV_OUTPUT_SIZE_1_IN_BYTES
#define CONV_OUTPUT_SIZE_2_IN_BYTES	CONV_INPUT_SIZE_2_IN_BYTES

// Conv3D_3
#define CONV_INPUT_SIZE_3_IN_BYTES	CONV_OUTPUT_SIZE_2_IN_BYTES
#define CONV_OUTPUT_SIZE_3_IN_BYTES	(CONV_INPUT_SIZE_3_IN_BYTES/2)

// Conv3D_4
#define CONV_INPUT_SIZE_4_IN_BYTES	CONV_OUTPUT_SIZE_3_IN_BYTES
#define CONV_OUTPUT_SIZE_4_IN_BYTES	CONV_INPUT_SIZE_4_IN_BYTES

// Conv3D_5
#define CONV_INPUT_SIZE_5_IN_BYTES	CONV_OUTPUT_SIZE_4_IN_BYTES
#define CONV_OUTPUT_SIZE_5_IN_BYTES	(CONV_INPUT_SIZE_5_IN_BYTES/10)

#define BUFFER_CONV_A_ADDR	0x200000
#define BUFFER_CONV_B_ADDR	0x400000
/* ********** */


/* AXI Lite */
#define DEV_ID_CONV		XPAR_CONV2D_0_DEVICE_ID
XConv2d AxiLiteConv;

#define DMA_DEV_ID_STREAM_CONV	XPAR_AXIDMA_0_DEVICE_ID
XAxiDma AxiDmaConv;
/* ******** */


// Initialize AXI lite device
int init_XAxiLiteConv_SimplePollMode(u16 deviceID) {
	XConv2d_Config* cfgPtr;
	int status;

	cfgPtr = XConv2d_LookupConfig(deviceID);
	if (!cfgPtr) {
		printf("No config found for device %d\r\n", deviceID);
		return XST_FAILURE;
	}

	status = XConv2d_CfgInitialize(&AxiLiteConv, cfgPtr);
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
int init_XAxiDmaConv_stream_SimplePollMode(u16 deviceID) {
	XAxiDma_Config* cfgPtr;
	int status;

	cfgPtr = XAxiDma_LookupConfig(deviceID);
	if (!cfgPtr) {
		printf("No config found for device %d\r\n", deviceID);
		return XST_FAILURE;
	}

	status = XAxiDma_CfgInitialize(&AxiDmaConv, cfgPtr);
	if (status != XST_SUCCESS) {
		printf("Initialization failed with status %d\r\n", status);
		return XST_FAILURE;
	}

	if (XAxiDma_HasSg(&AxiDmaConv)) {
		/**
		 * 1st test - it entered here.
		 * In vivado disable in the DMA: "Enable Scatter Gather Engine"
		 */
		printf("Device configured as SG mode \r\n");
		return XST_FAILURE;
	}

	/* Disable interrupts, we use polling mode	 */
	XAxiDma_IntrDisable(&AxiDmaConv, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&AxiDmaConv, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

	return XST_SUCCESS;
}

int init_conv3d() {
	int status;

	// AXI Lite
	status = init_XAxiLiteConv_SimplePollMode(DEV_ID_CONV);
	if (status != XST_SUCCESS) {
		printf("init_XAxiLiteConv_SimplePollMode: Failed\r\n");
		return XST_FAILURE;
	}

	// AXI Stream
	status = init_XAxiDmaConv_stream_SimplePollMode(DMA_DEV_ID_STREAM_CONV);
	if (status != XST_SUCCESS) {
		printf("init_XAxiDmaConv_stream_SimplePollMode: Failed\r\n");
		return XST_FAILURE;
	}

	return 0;
}

void HW_conv3d(unsigned int *input, unsigned int input_size, unsigned int *output, unsigned int output_size, int layerIndex) {
#ifdef DEBUG_CONV
	printf("START: HW_conv3d\r\n");
	printf("Sending layerIndex = %d (AXI Lite)\r\n", layerIndex);
#endif
	/* send config */
	XConv2d_Set_layerIndex(&AxiLiteConv, layerIndex);

#ifdef DEBUG_CONV
	printf("Conv2d_start\r\n");
#endif
	XConv2d_Start(&AxiLiteConv);

#ifdef DEBUG_CONV
	printf("SimpleTransfer: Rx\r\n");
#endif
	/* receive OUTPUT */
	int status;
	status = XAxiDma_SimpleTransfer(&AxiDmaConv,(UINTPTR)output, output_size, XAXIDMA_DEVICE_TO_DMA);
	if (status != XST_SUCCESS) {
		printf("Error: %d | Failed sending: output\n", status);
		return;
	}

#ifdef DEBUG_CONV
	printf("SimpleTransfer: Tx\r\n");
#endif
	/* send INPUT */
	status = XAxiDma_SimpleTransfer(&AxiDmaConv, (UINTPTR)input, input_size, XAXIDMA_DMA_TO_DEVICE);
	if (status != XST_SUCCESS) {
		printf("Error: %d | Failed sending: input\n", status);
		return;
	}

#ifdef DEBUG_CONV
	printf("Wait for Tx...\r\n");
#endif
	while (XAxiDma_Busy(&AxiDmaConv, XAXIDMA_DMA_TO_DEVICE)) { /* Wait for Tx*/ }
#ifdef DEBUG_CONV
	printf("Tx done.\r\n");

	printf("Wait for Rx...\r\n");
#endif
	while (XAxiDma_Busy(&AxiDmaConv, XAXIDMA_DEVICE_TO_DMA)) { /* Wait for Rx*/	}
#ifdef DEBUG_CONV
	printf("Rx done.\r\n");

	printf("END: HW_conv3d\r\n");
#endif
}

void conv3d_0(const unsigned char* input) {
	HW_conv3d(
			(unsigned int*)input, CONV_INPUT_SIZE_0_IN_BYTES,
			(unsigned int*)BUFFER_CONV_A_ADDR, CONV_OUTPUT_SIZE_0_IN_BYTES,
			0
	);
}
void conv3d_1() {
	HW_conv3d(
			(unsigned int*)BUFFER_CONV_A_ADDR, CONV_INPUT_SIZE_1_IN_BYTES,
			(unsigned int*)BUFFER_CONV_B_ADDR, CONV_OUTPUT_SIZE_1_IN_BYTES,
			1
	);
}
void conv3d_2() {
	HW_conv3d(
			(unsigned int*)BUFFER_CONV_B_ADDR, CONV_INPUT_SIZE_2_IN_BYTES,
			(unsigned int*)BUFFER_CONV_A_ADDR, CONV_OUTPUT_SIZE_2_IN_BYTES,
			2
	);
}
void conv3d_3() {
	HW_conv3d(
			(unsigned int*)BUFFER_CONV_A_ADDR, CONV_INPUT_SIZE_3_IN_BYTES,
			(unsigned int*)BUFFER_CONV_B_ADDR, CONV_OUTPUT_SIZE_3_IN_BYTES,
			3
	);
}
void conv3d_4() {
	HW_conv3d(
			(unsigned int*)BUFFER_CONV_B_ADDR, CONV_INPUT_SIZE_4_IN_BYTES,
			(unsigned int*)BUFFER_CONV_A_ADDR, CONV_OUTPUT_SIZE_4_IN_BYTES,
			4
	);
}
unsigned char* conv3d_5() {
	HW_conv3d(
			(unsigned int*)BUFFER_CONV_A_ADDR, CONV_INPUT_SIZE_5_IN_BYTES,
			(unsigned int*)BUFFER_CONV_B_ADDR, CONV_OUTPUT_SIZE_5_IN_BYTES,
			5
	);

	Xil_DCacheInvalidateRange((INTPTR)(BUFFER_CONV_B_ADDR), CONV_OUTPUT_SIZE_5_IN_BYTES);
	return (unsigned char*)BUFFER_CONV_B_ADDR;
}

#endif // CONV3D_H
