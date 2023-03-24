#include <ap_int.h>
#include "settings.h"


typedef ap_int<W_BIT_WIDTH> weigth_t;
typedef ap_uint<I_BIT_WIDTH> imap_t;
typedef ap_uint<I_BIT_WIDTH> omap_t;
typedef ap_int<B_BIT_WIDTH> bias_t;

typedef ap_int<16> count_t;
typedef ap_int<21> accum_t;  // I_BIT_WIDTH+W_BIT_WIDTH + 5


// conv2D_1, input/output with padding
void conv2D_1(
        imap_t input[IC2D_1_IHEIGHT * IC2D_1_IWIDTH * IC2D_1_ICHANNELS],
        weigth_t weights[IC2D_1_KSIZE * IC2D_1_KSIZE],
		bias_t bias[IC2D_1_BSIZE],
		omap_t output[IC2D_1_OHEIGHT * IC2D_1_OWIDTH * IC2D_1_OCHANNELS]
    ) {

    for (count_t orow = 1; orow < (IC2D_1_OHEIGHT-C2D_OFFSET); ++orow) {
        for (count_t ocol = C2D_OFFSET; ocol < (IC2D_1_OWIDTH-C2D_OFFSET); ++ocol) {
            accum_t acc = (accum_t) bias[0];
            omap_t acc_sat;
            for (count_t krow = 0; krow < IC2D_1_KSIZE; ++krow) {
                for (count_t kcol = 0; kcol < IC2D_1_KSIZE; ++kcol) {
                    count_t weight_1d_loc = (krow) * IC2D_1_KSIZE + (kcol);
                    count_t image_1d_loc = ((orow + krow-C2D_OFFSET) * IC2D_1_OWIDTH + (ocol + kcol-C2D_OFFSET));
                    acc += weights[weight_1d_loc] * input[image_1d_loc];
                    /*
                    printf("input=%d\n", input[image_1d_loc]);
                    printf("weights=%d\n", weights[weight_1d_loc]);
                    printf("acc=%d\n", acc);
                    */
                }
            }
            printf("\n");

            if (acc > 255)
                acc_sat = 255;
            else if (acc < 0)
                acc_sat = 0;    // ReLu
            else
                acc_sat = acc;
            output[(orow * IC2D_1_OWIDTH + ocol)] = acc_sat;
        }
    }
}


// The top-level function
void predict(
        imap_t input_values[IHEIGHT * IWIDTH * ICHANNELS],
        weigth_t weights[IC2D_1_KSIZE * IC2D_1_KSIZE],
        bias_t bias[IC2D_1_BSIZE],
        omap_t output[IC2D_1_IHEIGHT * IC2D_1_IWIDTH * IC2D_1_ICHANNELS]
        //omap_t output[OHEIGHT * OWIDTH * OCHANNELS]
    ) {

    imap_t input[IC2D_1_IHEIGHT * IC2D_1_IWIDTH * IC2D_1_ICHANNELS];

    // prepare
    for (count_t channel = 0; channel < IC2D_1_ICHANNELS; ++channel) {
        const int channel_offset = channel * IC2D_1_IWIDTH * IC2D_1_IHEIGHT;
        for (count_t orow = 0; orow < IC2D_1_IWIDTH; ++orow) {
            for (count_t ocol = 0; ocol < IC2D_1_IWIDTH; ++ocol) {
                const int index = (orow * IC2D_1_IWIDTH + ocol) + channel_offset;
                if (orow < C2D_OFFSET || orow >= C2D_OFFSET+IHEIGHT ||
                    ocol < C2D_OFFSET || ocol >= C2D_OFFSET+IWIDTH) {
                    input[index] = 0;
                }
                else {
                    accum_t value = input_values[(orow-C2D_OFFSET) * IWIDTH + (ocol-C2D_OFFSET)];
                    input[index] = value;
                }
            }
        }
    }

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
            width:            3

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
            width:            3

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
            width:            3

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
