#include <ap_int.h>


#define IHEIGHT 1
#define IWIDTH  40

#define CHANNELS 64

#define OHEIGHT 1
#define OWIDTH  5




#define K_SIZE  3

#define W_BIT_WIDTH 8
#define I_BIT_WIDTH 8

typedef ap_int<W_BIT_WIDTH> weigth_t;
typedef ap_uint<I_BIT_WIDTH> imap_t;
typedef ap_uint<I_BIT_WIDTH> omap_t;

typedef ap_int<32> bias_t;
typedef ap_int<13> count_t;
typedef ap_int<21> accum_t;  // I_BIT_WIDTH+W_BIT_WIDTH + 5

// The top-level function
void axil_conv2D0(imap_t img_in[IHEIGHT * IWIDTH],
		omap_t img_out[OHEIGHT * OWIDTH], weigth_t weights[K_SIZE * K_SIZE],
		bias_t bias) {
#pragma HLS INTERFACE s_axilite port=return bundle=BUS1
#pragma HLS INTERFACE s_axilite port=img_in bundle=BUS1
#pragma HLS INTERFACE s_axilite port=img_out bundle=BUS1
#pragma HLS INTERFACE s_axilite port=weights bundle=BUS1
#pragma HLS INTERFACE s_axilite port=bias bundle=BUS1

	loop_orow: for (count_t orow = 0; orow < OHEIGHT; orow++) {
		loop_ocol: for (count_t ocol = 0; ocol < OWIDTH; ocol++) {
			accum_t acc = (accum_t) bias;
			omap_t acc_sat;
			loop_k1: for (count_t krow = 0; krow < K_SIZE; krow++) {
				loop_k2: for (count_t kcol = 0; kcol < K_SIZE; kcol++) {
#pragma HLS PIPELINE
					count_t weight_1d_loc = (krow) * K_SIZE + (kcol);
					count_t image_1d_loc = (orow + krow) * IWIDTH + (ocol + kcol);
					acc += weights[weight_1d_loc] * img_in[image_1d_loc];
				}
			}
			if (acc > 255)
				acc_sat = 255;
			else if (acc < 0)
				acc_sat = 0;
			else
				acc_sat = acc;
			img_out[orow * OWIDTH + ocol] = acc_sat;
		}
	}
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


imap_t input[IHEIGHT * IWIDTH * CHANNELS];

// The top-level function
void predict(imap_t input_values[IHEIGHT * IWIDTH], omap_t output[OHEIGHT * OWIDTH * CHANNELS] /* temporary output */) {
    // prepare
    for count_t channel = 0; channel < CHANNELS; ++channel) {
        for (count_t orow = 0; orow < OHEIGHT; ++orow) {
            for (count_t ocol = 0; ocol < OWIDTH; ++ocol) {
                input[(orow * OWIDTH + ocol) * channel] = input_values[orow * OWIDTH + ocol];
            }
        }
    }

    conv2D();
    conv2D();
    maxpool2D();

    conv2D();
    conv2D();
    maxpool2D();

    conv2D();
    conv2D();
    maxpool2D();
}


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

