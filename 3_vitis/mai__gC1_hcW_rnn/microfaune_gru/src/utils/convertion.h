#ifndef CONVERTION_H
#define CONVERTION_H


#define ALIGN_TO_BYTE(value) (((value) + 7) & ~7)	// makes sure the size is multiple of 8 bytes


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
    	unsigned char gruC = input[i];
    	float gruF = fixed81ToFloat(input[i]);
    	//printf("%d > 0x%02x | %f\n", i, gruC, gruF);
        output[i] = gruF;
    }
}

#endif // CONVERTION_H
