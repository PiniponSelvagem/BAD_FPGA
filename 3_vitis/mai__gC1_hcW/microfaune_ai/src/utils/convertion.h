#ifndef CONVERTION_H
#define CONVERTION_H


#define ALIGN_TO_BYTE(value) (((value) + 7) & ~7)	// makes sure the size is multiple of 8 bytes

/**
 * Converts conv3d output to bgru input.
 * Convolution outputs:
 * > 4 bits
 * Bidirectional GRU accepts:
 * > 8 bits with 1 sign bit
 * To transform Conv to GRU the following structure of each 8 bit value much be achieved:
 * > [ SDDD DPPP ] or [ S DDDD PPP ]
 * - S -> sign bit
 * - D -> data bit from conv
 * - P -> padding bit
 * Since GRU accepts the Conv value without pre-processing other than expansion from 4 to 8 bits,
 * the structure of each 8 bit value can be simplified as:
 * > [ 0DDD D000 ] or [ 0 DDDD 000 ]
 * - 0 -> bit equal to zero
 * - D -> data bit unchanged, only shifted / adjusted
 *
 * input: output of conv3d_5()
 * inSizeBytes: size of output of conv3d_5(), in bytes
 * output: output used has input on bgru_0()
 *
 * NOTE: output must have double the size of input
 */
void conv2gru(unsigned char *input, int inSizeBytes, unsigned char* output) {
	for (int i=0, j=0; i<inSizeBytes; ++i) {
		unsigned char inVal = input[i];
		unsigned char inValL = (inVal & 0x0F) << 3;
		unsigned char inValH = (inVal & 0xF0) >> 1;
		output[j++] = inValL;
		output[j++] = inValH;
	}
}

/**
 * Converts fixed point 8 bits with 1 int to float.
 */
float fixed81ToFloat(unsigned char cVal) {
	int negSign = -1 ^ 0xFF;
	int value;
	if (cVal & 0x80)	// check if negative bit is active
		value = negSign | cVal;
	else
		value = cVal;

	return ((float)value/(float)(128));	// 127 = 2^7 --> 7 fractional bits
}

/**
 * Converts fixed point 8 bit array to float array
 * inSize: input/output array size
 */
void gru2td(unsigned char* input, int inSize, float* output) {
    for (int i=0; i<inSize; ++i) {
        output[i] = fixed81ToFloat(input[i]);
    }
}

#endif // CONVERTION_H
