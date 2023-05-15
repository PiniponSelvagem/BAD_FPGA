#include <ap_int.h>
#include "settings.h"


typedef ap_int<W_BIT_WIDTH> weigth_t;
typedef ap_int<I_BIT_WIDTH> imap_t;
typedef ap_int<I_BIT_WIDTH> omap_t;
typedef ap_int<B_BIT_WIDTH> bias_t;

typedef ap_int<16> count_t;
typedef ap_int<21> accum_t;  // I_BIT_WIDTH+W_BIT_WIDTH + 5


// conv2D_1, input/output with padding
void conv2D_1(
        imap_t input[C2D_1_IHEIGHT * C2D_1_IWIDTH * C2D_1_ICHANNELS],
        weigth_t weights[C2D_1_KSIZE * C2D_1_KSIZE * C2D_1_ICHANNELS],
        bias_t bias[C2D_1_BSIZE],
        omap_t output[C2D_1_OHEIGHT * C2D_1_OWIDTH * C2D_1_OCHANNELS]
    ) {

	/*
	 *
	 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	 * THIS IS NOT WORKING AS INTENDED
	 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	 *
	 *
	 */
    
    for (count_t channel = 0; channel < C2D_1_ICHANNELS; ++channel) {
        accum_t in_channel_offset  = channel * C2D_1_IWIDTH * C2D_1_IHEIGHT;
        accum_t out_channel_offset = channel * C2D_1_OWIDTH * C2D_1_OHEIGHT;
        accum_t weight_offset      = channel * C2D_1_KSIZE  * C2D_1_KSIZE;

        for (count_t orow = 1; orow < (C2D_1_OHEIGHT-C2D_OFFSET); ++orow) {
            for (count_t ocol = C2D_OFFSET; ocol < (C2D_1_OWIDTH-C2D_OFFSET); ++ocol) {
                accum_t acc = (accum_t) bias[channel];
                omap_t acc_sat;
                for (count_t krow = 0; krow < C2D_1_KSIZE; ++krow) {
                    for (count_t kcol = 0; kcol < C2D_1_KSIZE; ++kcol) {
                        count_t weight_1d_loc = ((krow) * C2D_1_KSIZE + (kcol)) + weight_offset;
                        count_t image_1d_loc = ((orow + krow-C2D_OFFSET) * C2D_1_IWIDTH + (ocol + kcol-C2D_OFFSET)) + in_channel_offset;
                        acc += weights[weight_1d_loc] * input[image_1d_loc];
                        /*
                        printf("input=%d\n", input[image_1d_loc]);
                        printf("weights=%d\n", weights[weight_1d_loc]);
                        printf("acc=%d\n", acc);
                        */
                    }
                }
                //printf("\n");

                if (acc > 255)
                    acc_sat = 255;
                else if (acc < 0)
                    acc_sat = 0;    // ReLu
                else
                    acc_sat = acc;
                output[(orow * C2D_1_OWIDTH + ocol) + out_channel_offset] = acc_sat;
            }
        }
    }
}

void maxpooling2D_1(
        imap_t input[MP2D_1_IHEIGHT * MP2D_1_IWIDTH * MP2D_1_CHANNELS],
        omap_t output[MP2D_1_OHEIGHT * MP2D_1_OWIDTH * MP2D_1_CHANNELS]
    ) {
    
    for (count_t channel = 0; channel < MP2D_1_CHANNELS; ++channel) {
        accum_t in_channel_offset  = channel * MP2D_1_IWIDTH * MP2D_1_IHEIGHT;
        accum_t out_channel_offset = channel * MP2D_1_OWIDTH * MP2D_1_OHEIGHT;

        for (count_t orow = 0; orow < MP2D_1_OHEIGHT; ++orow) {
            for (count_t ocol = 0; ocol < MP2D_1_OWIDTH; ++ocol) {
                count_t irow_start = orow * MP2D_1_HSTRIDE;
                count_t irow_end = irow_start + MP2D_1_HSTRIDE;

                count_t icol_start = ocol * (MP2D_1_WSTRIDE-1);
                count_t icol_end = icol_start + MP2D_1_WSTRIDE;

                omap_t maxval = (omap_t) MIN_VALUE_I;

                for (count_t irow = irow_start; irow < irow_end; ++irow) {
                    for (count_t icol = icol_start; icol < icol_end; ++icol) {
                        count_t image_1d_loc = ((orow + irow) * MP2D_1_IWIDTH + (ocol + icol)) + in_channel_offset;
                        omap_t val = input[image_1d_loc];
                        //printf(" %2d", val);
                        if (val > maxval)
                            maxval = val;
                    }
                }

                output[(orow * MP2D_1_OWIDTH + ocol) + out_channel_offset] = maxval;
            }
        }
    }
}


// The top-level function
void predict(
        imap_t input_values[IHEIGHT * IWIDTH * ICHANNELS],
        weigth_t weights[C2D_1_KSIZE * C2D_1_KSIZE * C2D_1_ICHANNELS],
        bias_t bias[C2D_1_BSIZE],
        omap_t output[C2D_1_OHEIGHT * C2D_1_OWIDTH * C2D_1_OCHANNELS]
        //omap_t output[OHEIGHT * OWIDTH * OCHANNELS]
    ) {

    imap_t input[C2D_1_IHEIGHT * C2D_1_IWIDTH * C2D_1_ICHANNELS];

    // prepare
    for (count_t channel = 0; channel < C2D_1_ICHANNELS; ++channel) {
        count_t channel_offset = channel * C2D_1_IWIDTH * C2D_1_IHEIGHT;
        for (count_t orow = 0; orow < C2D_1_IHEIGHT; ++orow) {
            for (count_t ocol = 0; ocol < C2D_1_IWIDTH; ++ocol) {
                count_t index = (orow * C2D_1_IWIDTH + ocol) + channel_offset;
                if (orow < C2D_OFFSET || orow >= C2D_OFFSET+IHEIGHT ||
                    ocol < C2D_OFFSET || ocol >= C2D_OFFSET+IWIDTH) {
                    input[index] = 0;
                }
                else {
                    accum_t value = input_values[(orow-C2D_OFFSET) * IWIDTH + (ocol-C2D_OFFSET)];
                    input[index] = value;
                    //printf("c %2d -> %2d -> %d\n", channel, input[index], index);
                }
            }
        }
    }

    /*
    accum_t i = 0, j = 0;
    for (count_t channel = 0; channel < IC2D_1_ICHANNELS; ++channel) {
        accum_t channel_offset  = channel * IC2D_1_IWIDTH * IC2D_1_IHEIGHT;
        for (i = 0; i < IC2D_1_IHEIGHT; ++i) {
			for (j = 0; j < IC2D_1_IWIDTH; ++j) {
				printf("%2d ", input[(i * IC2D_1_IWIDTH + j) + channel_offset]);
			}
			printf("\n");
		}
        printf("\n");
    }
    */

    conv2D_1(input, weights, bias, output);
    /*
    conv2D();
    maxpool2D();

    conv2D();
    conv2D();
    maxpool2D();

    conv2D();
    conv2D();
    maxpool2D();
    */
}

// CNN - START/////////////////////////////////
/* conv2D - 1
    input:  (1*1*40*1)
            batch_size:       1
            height:           1
            width:           40
            number_channels:  1

    filter: (64*3*3*1)
            batch_size:      64
            height:           3
            width:            3
            number_channels:  1

    bias:   (64)

    output: (1*1*40*64)
            batch_size:       1
            height:           1
            width:           40
            number_channels: 64
*/
/* conv2D - 2
    input:  (1*1*40*64)
            batch_size:       1
            height:           1
            width:           40
            number_channels: 64

    filter: (64*3*3*64)
            batch_size:      64
            height:           3
            width:            3
            number_channels: 64

    bias:   (64)

    output: (1*1*40*64)
            batch_size:       1
            height:           1
            width:           40
            number_channels: 64
*/
/* maxpool2D - 1
    input:  (1*1*40*64)
            batch_size:       1
            height:           1
            width:           40
            number_channels: 64

    filter:
            height:           1
            width:            2

    output: (1*1*20*64)
            batch_size:       1
            height:           1
            width:           20
            number_channels: 64
*/

/* conv2D - 3
    input:  (1*1*20*64)
            batch_size:       1
            height:           1
            width:           20
            number_channels: 64

    filter: (64*3*3*64)
            batch_size:      64
            height:           3
            width:            3
            number_channels: 64

    bias:   (64)

    output: (1*1*20*64)
            batch_size:       1
            height:           1
            width:           20
            number_channels: 64
*/
/* conv2D - 4
    input:  (1*1*20*64)
            batch_size:       1
            height:           1
            width:           20
            number_channels: 64

    filter: (64*3*3*64)
            batch_size:      64
            height:           3
            width:            3
            number_channels: 64

    bias:   (64)

    output: (1*1*20*64)
            batch_size:       1
            height:           1
            width:           20
            number_channels: 64
*/
/* maxpool2D - 2
    input:  (1*1*20*64)
            batch_size:       1
            height:           1
            width:           20
            number_channels: 64

    filter:
            height:           1
            width:            2

    output: (1*1*10*64)
            batch_size:       1
            height:           1
            width:           10
            number_channels: 64
*/

/* conv2D - 5
    input:  (1*1*10*64)
            batch_size:       1
            height:           1
            width:           10
            number_channels: 64

    filter: (64*3*3*64)
            batch_size:      64
            height:           3
            width:            3
            number_channels: 64

    bias:   (64)

    output: (1*1*10*64)
            batch_size:       1
            height:           1
            width:           10
            number_channels: 64
*/
/* conv2D - 6
    input:  (1*1*10*64)
            batch_size:       1
            height:           1
            width:           10
            number_channels: 64

    filter: (64*3*3*64)
            batch_size:      64
            height:           3
            width:            3
            number_channels: 64

    bias:   (64)

    output: (1*1*10*64)
            batch_size:       1
            height:           1
            width:           10
            number_channels: 64
*/
/* maxpool2D - 3
    input:  (1*1*10*64)
            batch_size:       1
            height:           1
            width:           10
            number_channels: 64

    filter:
            height:           1
            width:            2

    output: (1*1*5*64)
            batch_size:       1
            height:           1
            width:            5
            number_channels: 64
*/
// CNN - END //////////////////////////////////


// RNN - START/////////////////////////////////
// ???
// RNN - END //////////////////////////////////




/*
void conv2D(imap_t img_in[IHEIGHT * IWIDTH * CHANNELS]) {
	for (count_t orow = 0; orow < OHEIGHT; ++orow) {
		for (count_t ocol = 0; ocol < OWIDTH; ++ocol) {
			accum_t acc = (accum_t) bias;
			omap_t acc_sat;
			for (count_t krow = 0; krow < K_SIZE; ++krow) {
				for (count_t kcol = 0; kcol < K_SIZE; ++kcol) {
					count_t weight_1d_loc = (krow) * K_SIZE + (kcol);
					count_t image_1d_loc = (orow + krow) * IWIDTH + (ocol + kcol);
					acc += weights[weight_1d_loc] * img_in[image_1d_loc];
				}
			}
			if (acc > 255)
				acc_sat = 255;
			else if (acc < 0)
				acc_sat = 0;    // ReLu
			else
				acc_sat = acc;
			img_out[orow * OWIDTH + ocol] = acc_sat;
		}
	}
}
*/
