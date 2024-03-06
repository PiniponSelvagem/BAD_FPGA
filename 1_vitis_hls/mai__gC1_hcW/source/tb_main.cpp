#define HW_IP 1
#include "hls_stream.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>

#include "types.h"
#include "size_conv3D.h"
#include "size_bgru.h"

#include "loader.h"

#include "soft_timedist.h"
#include "soft_reducemax.h"

#include "utils.h"


//#define PRINT_STATS
//#define DEBUG_CONV
#define DO_CNN
#define DO_RNN
#define VALIDATE_OUTPUT


typedef ap_axis<64, 0, 0, 0> in_pkt;
typedef ap_axis<64, 0, 0, 0> out_pkt;

hls::stream<in_pkt> strm_in;
hls::stream<out_pkt> strm_out;


void conv2D(hls::stream<in_pkt> &strm_in, hls::stream<out_pkt> &strm_out, int layerIndex);
void gru(hls::stream<in_pkt> &strm_in, hls::stream<out_pkt> &strm_out, int layerIndex);

imap_t input_0[IHEIGHT*IWIDTH/PACKET_CNN];
imap_t input_1[IHEIGHT*IWIDTH*CHANNELS/PACKET_CNN];
imap_t input_2[IHEIGHT*IWIDTH_1*CHANNELS/PACKET_CNN];
imap_t input_3[IHEIGHT*IWIDTH_1*CHANNELS/PACKET_CNN];
imap_t input_4[IHEIGHT*IWIDTH_2*CHANNELS/PACKET_CNN];
float input_gru[IHEIGHT*FILTERS];

float outputConv_float[IHEIGHT*FILTERS];
gru_imap_t outputConv_converted[IHEIGHT*FILTERS];
gru_omap_t outputGRU0[IHEIGHT*(GRU_FILTERS*2)];
gru_omap_t outputGRU1[IHEIGHT*(GRU_FILTERS*2)];
float outputGRU1_float[IHEIGHT*(GRU_FILTERS*2)];

float outputTD0[IHEIGHT*FILTERS];
float outputLS[IHEIGHT];
float outputGS[1];

/* output expected validation */
float output_expect_GRU0[IHEIGHT*(GRU_FILTERS*2)];
float output_expect_GRU1[IHEIGHT*(GRU_FILTERS*2)];
float output_expect_LS[IHEIGHT];
float output_expect_GS[1];


float td0_kernel[FILTERS*(FILTERS*2)];
float td0_bias[FILTERS];

float td1_kernel[1*FILTERS];
float td1_bias[1];

void writeInput(imap_t* input) {
	in_pkt tmp;
	int bytesTransfered = 0;
    for (int i=0, j=0, i_pad=0; i<IHEIGHT*IWIDTH*CHANNELS/PACKET_CNN; i++) {
		ap_uint<6> rangeStart = (j % PACKET_CNN) * 4;
		ap_uint<6> rangeEnd   = rangeStart + 3;
		//printf("i %d, j %d, rs %d, re %d | jP %f\n", i, j, (int)rangeStart, (int)rangeEnd, (float)(j/PACKET_CNN));
		if (i_pad == 0) {
			tmp.data = input[j/PACKET_CNN].range(rangeEnd,rangeStart);
			++j;
		}
		else
			tmp.data = 0x0000000000000000;
		++i_pad;
		if (i_pad == 4) i_pad = 0;
		//printf("writeInput[%d] = 0x%0x 0x%0x\n", i, (int)tmp.data.range(63,32), (int)tmp.data.range(31,0));
        if (i == (IHEIGHT*IWIDTH*CHANNELS/PACKET_CNN-1)) tmp.last = (ap_int<1>)1;
        else tmp.last = (ap_int<1>)0;
        strm_in.write(tmp);
        bytesTransfered += 8;	// 64 bits has 8 bytes
    }
    printf("Total bytes transfered = %d\n", bytesTransfered);
}
void writeInputNextLayer() {
	int i = 0;
    in_pkt tmp;
    out_pkt tmpo;
    int bytesTransfered = 0;
	typedef ap_ufixed<4,0> outputF;
    printf(" --- output of previous layer ---\n");
    while (!strm_out.empty()) {
        tmpo = strm_out.read();
        bytesTransfered += 8;	// 64 bits has 8 bytes
        int out0 = (int)tmpo.data.range(3,0);	outputF out0F; out0F.range(3,0) = tmpo.data.range(3,0);
        int out1 = (int)tmpo.data.range(7,4);	outputF out1F; out1F.range(3,0) = tmpo.data.range(7,4);
        int out2 = (int)tmpo.data.range(11,8);  outputF out2F; out2F.range(3,0) = tmpo.data.range(11,8);
        int out3 = (int)tmpo.data.range(15,12); outputF out3F; out3F.range(3,0) = tmpo.data.range(15,12);

        int out4 = (int)tmpo.data.range(19,16); outputF out4F; out4F.range(3,0) = tmpo.data.range(19,16);
        int out5 = (int)tmpo.data.range(23,20); outputF out5F; out5F.range(3,0) = tmpo.data.range(23,20);
        int out6 = (int)tmpo.data.range(27,24); outputF out6F; out6F.range(3,0) = tmpo.data.range(27,24);
        int out7 = (int)tmpo.data.range(31,28); outputF out7F; out7F.range(3,0) = tmpo.data.range(31,28);

        int out8 = (int)tmpo.data.range(35,32); outputF out8F; out8F.range(3,0) = tmpo.data.range(35,32);
        int out9 = (int)tmpo.data.range(39,36); outputF out9F; out9F.range(3,0) = tmpo.data.range(39,36);
        int outA = (int)tmpo.data.range(43,40); outputF outAF; outAF.range(3,0) = tmpo.data.range(43,40);
        int outB = (int)tmpo.data.range(47,44); outputF outBF; outBF.range(3,0) = tmpo.data.range(47,44);

        int outC = (int)tmpo.data.range(51,48); outputF outCF; outCF.range(3,0) = tmpo.data.range(51,48);
        int outD = (int)tmpo.data.range(55,52); outputF outDF; outDF.range(3,0) = tmpo.data.range(55,52);
        int outE = (int)tmpo.data.range(59,56); outputF outEF; outEF.range(3,0) = tmpo.data.range(59,56);
        int outF = (int)tmpo.data.range(63,60); outputF outFF; outFF.range(3,0) = tmpo.data.range(63,60);

        printf("%02d - hex %x%x%x%x %x%x%x%x %x%x%x%x %x%x%x%x   |   result %2d, %2d, %2d, %2d,   %2d, %2d, %2d, %2d,   %2d, %2d, %2d, %2d,   %2d, %2d, %2d, %2d   "
				"|   %f, %f, %f, %f,   %f, %f, %f, %f,   %f, %f, %f, %f,   %f, %f, %f, %f\n", i++,
				//
				outF, outE, outD, outC,
				outB, outA, out9, out8,
				out7, out6, out5, out4,
				out3, out2, out1, out0,
				//
				out0, out1, out2, out3,
				out4, out5, out6, out7,
				out8, out9, outA, outB,
				outC, outD, outE, outF,
				(float)out0F, (float)out1F, (float)out2F, (float)out3F,
				(float)out4F, (float)out5F, (float)out6F, (float)out7F,
				(float)out8F, (float)out9F, (float)outAF, (float)outBF,
				(float)outCF, (float)outDF, (float)outEF, (float)outFF
		);
    	tmp = tmpo;
    	if (!strm_out.empty())
    		tmp.last = (ap_int<1>)0;
    	else
    		tmp.last = (ap_int<1>)1;
    	strm_in.write(tmp);
    }
    printf("Total bytes transfered = %d\n", bytesTransfered);
}
void printLastLayerOutput() {
	int i = 0;
    out_pkt tmpo;
	typedef ap_ufixed<4,0> outputF;
	int bytesTransfered = 0;
    while (!strm_out.empty()) {
        tmpo = strm_out.read();
        bytesTransfered += 8;	// 64 bits has 8 bytes
        int out0 = (int)tmpo.data.range(3,0);	outputF out0F; out0F.range(3,0) = tmpo.data.range(3,0);
        int out1 = (int)tmpo.data.range(7,4);	outputF out1F; out1F.range(3,0) = tmpo.data.range(7,4);
        int out2 = (int)tmpo.data.range(11,8);  outputF out2F; out2F.range(3,0) = tmpo.data.range(11,8);
        int out3 = (int)tmpo.data.range(15,12); outputF out3F; out3F.range(3,0) = tmpo.data.range(15,12);

        int out4 = (int)tmpo.data.range(19,16); outputF out4F; out4F.range(3,0) = tmpo.data.range(19,16);
        int out5 = (int)tmpo.data.range(23,20); outputF out5F; out5F.range(3,0) = tmpo.data.range(23,20);
        int out6 = (int)tmpo.data.range(27,24); outputF out6F; out6F.range(3,0) = tmpo.data.range(27,24);
        int out7 = (int)tmpo.data.range(31,28); outputF out7F; out7F.range(3,0) = tmpo.data.range(31,28);

        int out8 = (int)tmpo.data.range(35,32); outputF out8F; out8F.range(3,0) = tmpo.data.range(35,32);
        int out9 = (int)tmpo.data.range(39,36); outputF out9F; out9F.range(3,0) = tmpo.data.range(39,36);
        int outA = (int)tmpo.data.range(43,40); outputF outAF; outAF.range(3,0) = tmpo.data.range(43,40);
        int outB = (int)tmpo.data.range(47,44); outputF outBF; outBF.range(3,0) = tmpo.data.range(47,44);

        int outC = (int)tmpo.data.range(51,48); outputF outCF; outCF.range(3,0) = tmpo.data.range(51,48);
        int outD = (int)tmpo.data.range(55,52); outputF outDF; outDF.range(3,0) = tmpo.data.range(55,52);
        int outE = (int)tmpo.data.range(59,56); outputF outEF; outEF.range(3,0) = tmpo.data.range(59,56);
        int outF = (int)tmpo.data.range(63,60); outputF outFF; outFF.range(3,0) = tmpo.data.range(63,60);

        printf("%02d - hex %x%x%x%x %x%x%x%x %x%x%x%x %x%x%x%x   |   result %2d, %2d, %2d, %2d,   %2d, %2d, %2d, %2d,   %2d, %2d, %2d, %2d,   %2d, %2d, %2d, %2d   "
				"|   %f, %f, %f, %f,   %f, %f, %f, %f,   %f, %f, %f, %f,   %f, %f, %f, %f\n", i++,
				//
				outF, outE, outD, outC,
				outB, outA, out9, out8,
				out7, out6, out5, out4,
				out3, out2, out1, out0,
				//
				out0, out1, out2, out3,
				out4, out5, out6, out7,
				out8, out9, outA, outB,
				outC, outD, outE, outF,
				(float)out0F, (float)out1F, (float)out2F, (float)out3F,
				(float)out4F, (float)out5F, (float)out6F, (float)out7F,
				(float)out8F, (float)out9F, (float)outAF, (float)outBF,
				(float)outCF, (float)outDF, (float)outEF, (float)outFF
		);
    }
    printf("Total bytes transfered = %d\n", bytesTransfered);
}

void writeWeigthGRU(gru_weigth_t* weigth, size_t size) {
	in_pkt tmp;
    tmp.last = (ap_int<1>)0;
    for (int i=0; i<size; ) {
        tmp.data.range(7,0)   = (int)weigth[i++].range(7,0);
        tmp.data.range(15,8)  = (int)weigth[i++].range(7,0);
        tmp.data.range(23,16) = (int)weigth[i++].range(7,0);
        tmp.data.range(31,24) = (int)weigth[i++].range(7,0);

        tmp.data.range(39,32) = (int)weigth[i++].range(7,0);
        tmp.data.range(47,40) = (int)weigth[i++].range(7,0);
        tmp.data.range(55,48) = (int)weigth[i++].range(7,0);
        tmp.data.range(63,56) = (int)weigth[i++].range(7,0);
        strm_in.write(tmp);
    }
}
void conv2gru() {
    /*
     * Currently this function saves the output of last CONV to an array.
     * To send the output to GRU, writeInputNextGRU function should be used.
     */

	int i = 0;
	in_pkt tmp;     // to send to next layer
	out_pkt tmpo;   // to receive from prev layer
	typedef ap_ufixed<4,0> outputF;
    typedef ap_fixed<8,1> inputF;
	printf(" --- conv2gru ---\n");

#ifndef DO_CONV
    printf("INFO: DO_CONV is not defined. Filling output stream for GRU input...\n");
    int cycles = 0;
    while(i<IHEIGHT*FILTERS) {
        outputF out0 = outputConv_float[i++];     tmpo.data.range(3,0)   = out0.range(3,0);
        outputF out1 = outputConv_float[i++];     tmpo.data.range(7,4)   = out1.range(3,0);
        outputF out2 = outputConv_float[i++];     tmpo.data.range(11,8)  = out2.range(3,0);
        outputF out3 = outputConv_float[i++];     tmpo.data.range(15,12) = out3.range(3,0);

        outputF out4 = outputConv_float[i++];     tmpo.data.range(19,16) = out4.range(3,0);
        outputF out5 = outputConv_float[i++];     tmpo.data.range(23,20) = out5.range(3,0);
        outputF out6 = outputConv_float[i++];     tmpo.data.range(27,24) = out6.range(3,0);
        outputF out7 = outputConv_float[i++];     tmpo.data.range(31,28) = out7.range(3,0);

        outputF out8 = outputConv_float[i++];     tmpo.data.range(35,32) = out8.range(3,0);
        outputF out9 = outputConv_float[i++];     tmpo.data.range(39,36) = out9.range(3,0);
        outputF outA = outputConv_float[i++];     tmpo.data.range(43,40) = outA.range(3,0);
        outputF outB = outputConv_float[i++];     tmpo.data.range(47,44) = outB.range(3,0);

        outputF outC = outputConv_float[i++];     tmpo.data.range(51,48) = outC.range(3,0);
        outputF outD = outputConv_float[i++];     tmpo.data.range(55,52) = outD.range(3,0);
        outputF outE = outputConv_float[i++];     tmpo.data.range(59,56) = outE.range(3,0);
        outputF outF = outputConv_float[i++];     tmpo.data.range(63,60) = outF.range(3,0);

        if (i+1 < IHEIGHT*FILTERS)
            tmpo.last = (ap_int<1>)0;
        else
            tmpo.last = (ap_int<1>)1;
        strm_out.write(tmpo);

        ++cycles;
    }
    printf("INFO: Stream filled in %d cycles.\n", cycles);
    i = 0;  // reset i to initial value
#endif

	while (!strm_out.empty()) {
        /*
        conv -> 4 bits 0 int
        gru  -> 8 bits 1 int
        --------------------
        gru bits convertion conv2gru:
        S -> sign == 0
        C -> conv result
        0 -> padding
        [ SCCC C000 ]
        */
		tmpo = strm_out.read();

        int out_offset = 64/2;   // because PACKET_CNN == 16, PACKET_GRU == 8, and that means 2 reads must be made for the convertion
        int out_databits = 64/PACKET_CNN;
        int in_databits = 64/PACKET_GRU;
        for (int section = 0; section < 2; ++section) {
            for (int subOffset = 0; subOffset < 64/PACKET_GRU; ++subOffset) {
                int startPad  = 0;   int endPad  = 2;   // padding
                int startData = 3;   int endData = 6;   // conv result
                int startSign = 7;   int endSign = 7;   // sign bit always positive
                startPad  += in_databits * subOffset;  endPad  += in_databits * subOffset;
                startData += in_databits * subOffset;  endData += in_databits * subOffset;
                startSign += in_databits * subOffset;  endSign += in_databits * subOffset;

                int startOutData = 0;   int endOutData = 3;         // conv result data
                startOutData += out_databits * subOffset + (out_offset * section);
                endOutData   += out_databits * subOffset + (out_offset * section);

                tmp.data.range(endPad, startPad)  = (int)0;
                tmp.data.range(endData,startData) = (int)tmpo.data.range(endOutData,startOutData);
                tmp.data.range(endSign,startSign) = (int)0;

                outputConv_converted[i++].range(7,0) = tmp.data.range(endSign,startPad);
                printf("outputConv_converted[%d] = 0x%02x\n", i, (int)outputConv_converted[i-1].range(7,0));
            }
        }
	}
}
void flipArrayLines(gru_imap_t* array, size_t size, int nCols) {
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
        	gru_imap_t temp = array[start + i];
            array[start + i] = array[end + i];
            array[end + i] = temp;
        }

        // Move indices towards the center for the next row
        start += nCols;
        end -= nCols;
    }
}
void writeInputNextGRU(gru_omap_t* outputGRU, int nCols) {
	in_pkt tmp;     // to send to next layer

    int idx;
    int bytesTransfered = 0;
    for (int i = 0; i < IHEIGHT*nCols; ) {
        for (int subOffset = 0; subOffset < 64/PACKET_GRU; ++subOffset) {
			int dataBits = 8;
			int startData = 0;  int endData = dataBits-1;
			startData += dataBits * subOffset;  endData += dataBits * subOffset;

			tmp.data.range(endData,startData) = outputGRU[i++].range(7,0);
		}
		if (i<IHEIGHT-1)
			tmp.last = (ap_int<1>)0;
		else
			tmp.last = (ap_int<1>)1;
		strm_in.write(tmp);
		bytesTransfered += 8;	// 64 bits has 8 bytes
    }
    printf("writeInputNextGRU: Total bytes transfered = %d\n", bytesTransfered);
}
void readOutputGRU(const int nFilters, int isForward, gru_omap_t* output) {
    out_pkt tmpo;
    const int databits = 64/PACKET_GRU;
#define GRU_N_ELEMS_SIZE    (GRU_IN_LINES*(nFilters*2))

    int i=0;
    if (!isForward)
        i += GRU_N_ELEMS_SIZE-1;     // since this is BACKWARD, adjust the start offset to the end to do backwards fill

    int nDataRead = 0;
    int nColsRead = 0;
    int exit = 0;
    int bytesTransfered = 0;
    while (!strm_out.empty()) {
		tmpo = strm_out.read();
		bytesTransfered += 8;	// 64 bits has 8 bytes
        if (exit)
            continue;

        for (int subOffset = 0; subOffset < 64/PACKET_GRU; ++subOffset) {
            int start = databits * subOffset;
            int end   = (databits * subOffset) + (PACKET_GRU-1);

            output[i].range(7,0) = tmpo.data.range(end, start);
            isForward ? ++i : --i;
            ++nColsRead;

            if (++nDataRead >= GRU_N_ELEMS_SIZE) {  // x2 because it is BiGRU and we only receive part of the input
                exit = 1;  // just in case so it dosent write outside of output array
                break;
            }
            
            if (nColsRead>=nFilters) {
                isForward ? i += nFilters : i -= nFilters; // offset due to the other part of the output of the other direction 
                nColsRead = 0;
            }
        }
	}
    printf("readOutputGRU: Total bytes transfered = %d\n", bytesTransfered);
}
void printGRUoutput(gru_omap_t* output) {
	#define OUT_WIDTH_GRU (GRU_FILTERS*2)
	for (int i=0; i<IHEIGHT; ++i) {
		printf("[%3d] - ", i);
		for (int j=0; j<OUT_WIDTH_GRU; ++j) {
			printf("0x%02x %10f | ", output[(i*OUT_WIDTH_GRU)+j], output[(i*OUT_WIDTH_GRU)+j].to_float());
		}
		printf("\n");
    }
}

void gru2td() {
    printf(" --- gru2td ---\n");
    for (int i=0; i<IHEIGHT*(GRU_FILTERS*2); ++i) {
    	unsigned char gruC = outputGRU1[i].range(7,0);
    	float gruF = outputGRU1[i].to_float();
    	printf("%d > 0x%02x | %f\n", i, gruC, gruF);
        outputGRU1_float[i] = gruF;
    }
}

// INFO: Currently this method can only compare float arrays!
void compareStats(char* msg, float* actual, float* expected, int size, int breakAfter) {
	printf(msg);
	float* pActual = actual;
	float* pExpected = expected;
	for (int i=0; i<size; ++i) {
		float vActual = *pActual;
		float vExpected = *pExpected;
		float change = vExpected - vActual;
		++pActual;
		++pExpected;

		if (i<breakAfter)
			printf("%12f | %12f | %12f\n", vActual, change, vExpected);
		else {
            printf("     ...     |      ...     |      ...\n");
			break;
        }
	}
}
void compareStats_APF(char* msg, gru_omap_t* actual, float* expected, int size, int breakAfter) {
	printf(msg);
	gru_omap_t* pActual = actual;
	float* pExpected = expected;
	for (int i=0; i<size; ++i) {
		gru_omap_t vActual = *pActual;
		float vExpected = *pExpected;
		float change = vExpected - vActual.to_float();
		++pActual;
		++pExpected;

		if (i<breakAfter)
			printf("%12f | %12f | %12f\n", vActual.to_float(), change, vExpected);
		else {
            printf("     ...     |      ...     |      ...\n");
			break;
        }
	}
}


int main() {
	loadIO(
		input_0,
		outputConv_float,
		output_expect_GRU0,
		output_expect_GRU1,
		output_expect_LS,
		output_expect_GS
	);
    
    loadWeights(
		td0_kernel, td0_bias,
		td1_kernel, td1_bias
	);
    //loadSigmoidTable();
    //loadTanhTable();

    /*
    printf("\ninput_0:\n");
    for (int idx=0; idx<IHEIGHT*IWIDTH*CHANNELS/PACKET_CNN; idx++) {
    	printf("idx=%d | 0x%016llx\n", idx, input_0[idx]);
    }
	*/

#ifdef DO_CNN
    printf("CONV_0\n");
    writeInput(input_0);
#if HW_IP
    conv2D(strm_in, strm_out, 0);
#endif

    printf("CONV_1\n");
    writeInputNextLayer();
#if HW_IP
    conv2D(strm_in, strm_out, 1);
#endif

    printf("CONV_2\n");
    writeInputNextLayer();
#if HW_IP
    conv2D(strm_in, strm_out, 2);
#endif

    printf("CONV_3\n");
    writeInputNextLayer();
#if HW_IP
    conv2D(strm_in, strm_out, 3);
#endif

    printf("CONV_4\n");
    writeInputNextLayer();
#if HW_IP
    conv2D(strm_in, strm_out, 4);
#endif

    printf("CONV_5\n");
    writeInputNextLayer();
#if HW_IP
    conv2D(strm_in, strm_out, 5);
#endif

#ifndef DO_RNN
    printLastLayerOutput();
#else
    printf("INFO: CONV_5 print not displayed. To show that, implement a print in writeInputNextGRU.");
#endif

#endif	// !DO_CONV

#ifdef DO_RNN
    conv2gru();

    /*
    for (int i=0; i<431*64; ++i) {
        printf("[%d] = %f", i, outputConv_converted[i].to_float());
        printf("  ---  {%d}-> %f\n", i, outputConv_float[i]);
    }
	*/

	printf("###################################### GRU_0 ######################################\n");
	printf("---- 0 FORWARD ----\n");
    writeInputNextGRU(outputConv_converted, FILTERS);
	gru( // GRU_0_F
		strm_in,
		strm_out,
		0
	);
    readOutputGRU(GRU_FILTERS, GRU_FORWARD, outputGRU0);
    printGRUoutput(outputGRU0);

	printf("---- 0 BACKWARD ----\n");
	flipArrayLines(outputConv_converted, IHEIGHT*FILTERS, FILTERS);
    writeInputNextGRU(outputConv_converted, FILTERS);     // send again same output of conv as input flipped
	gru( // GRU_0_B
		strm_in,
		strm_out,
		1
	);
    readOutputGRU(GRU_FILTERS, GRU_BACKWARD, outputGRU0);
	printGRUoutput(outputGRU0);

	printf("\n\n\n###################################### GRU_1 ######################################\n");
	printf("---- 1 FORWARD ----\n");
    writeInputNextGRU(outputGRU0, GRU_FILTERS*2);
	gru( // GRU_1_F
		strm_in,
		strm_out,
		2
	);
    readOutputGRU(GRU_FILTERS, GRU_FORWARD, outputGRU1);
    printGRUoutput(outputGRU1);
    
	printf("---- 1 BACKWARD ----\n");
	flipArrayLines(outputGRU0, IHEIGHT*GRU_FILTERS*2, GRU_FILTERS*2);
    writeInputNextGRU(outputGRU0, GRU_FILTERS*2);
	gru( // GRU_1_B
		strm_in,
		strm_out,
		3
	);
    readOutputGRU(GRU_FILTERS, GRU_BACKWARD, outputGRU1);
	printGRUoutput(outputGRU1);



	printf("\n\n\n");
	gru2td();

	printf("\n\n\n###################################### TD_0 ######################################\n");
    timedistributed_dense( // TDIST_0 + Dense
    	GRU_FILTERS*2,
        FILTERS, GRU_FILTERS*2,
        TD_0__OUT_COLS,
		outputGRU1_float,
		td0_kernel,
		td0_bias,
        outputTD0
    );

	for (int i=0; i<IHEIGHT; ++i) {
		printf("[%3d] - ", i);
		for (int j=0; j<TD_0__OUT_COLS; ++j) {
			printf("%10f ", outputTD0[(i*TD_0__OUT_COLS)+j]);
		}
		printf("\n");
	}

	printf("\n\n\n###################################### TD_1 ######################################\n");
    timedistributed_dense( // TDIST_1 + Dense
    	TD_0__OUT_COLS,
        1, FILTERS,
        1,
		outputTD0,
		td1_kernel,
		td1_bias,
		outputLS   // LOCAL OUTPUT
    );

	for (int i=0; i<IHEIGHT; ++i) {
		printf("[%3d] - ", i);
		printf("%10f ", outputLS[i]);
		printf("\n");
	}

	printf("\n\n\n###################################### R_MX ######################################\n");
    reducemax_1( // RMAX_1
        outputLS,    // LOCAL OUTPUT -> 431
        outputGS     // GLOBAL OUTPUT -> 1
    );

    printf("GS = %f\n", outputGS[0]);

#ifdef VALIDATE_OUTPUT
    printf("\n\n\n#### #### #### ####\n");
#define BREAK_AFTER 8
    compareStats_APF("Stats GRU_0: (actual | difference | expected)\n", outputGRU0, output_expect_GRU0, IHEIGHT*(GRU_FILTERS*2), BREAK_AFTER);
    compareStats_APF("Stats GRU_1: (actual | difference | expected)\n", outputGRU1, output_expect_GRU1, IHEIGHT*(GRU_FILTERS*2), BREAK_AFTER);
    compareStats("Stats LS: (actual | difference | expected)\n", outputLS, output_expect_LS, IHEIGHT, BREAK_AFTER);
    compareStats("Stats GS: (actual | difference | expected)\n", outputGS, output_expect_GS, 1, BREAK_AFTER);
#endif
#endif // DO_GRU
    return 0;
}
