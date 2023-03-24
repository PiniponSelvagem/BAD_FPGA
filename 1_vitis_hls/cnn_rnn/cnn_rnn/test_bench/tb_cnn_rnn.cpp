#include <ap_int.h>
#include "../Source/settings.h"

static unsigned char img_in[IHEIGHT * IWIDTH] = {
	//1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};
static signed char kernel[IC2D_1_KSIZE * IC2D_1_KSIZE] = {
    0, 0, 0,
    0, 1, 0,
    0, 0, 0,
};
static signed char bias[IC2D_1_BSIZE] = {
	 1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
	11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
	21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
	31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
	41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
	51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
	61, 62, 63, 64,
};

static unsigned char img_out[IC2D_1_IHEIGHT * IC2D_1_IWIDTH * IC2D_1_ICHANNELS];

typedef ap_int<W_BIT_WIDTH> weigth_t;
typedef ap_uint<I_BIT_WIDTH> imap_t;
typedef ap_uint<I_BIT_WIDTH> omap_t;
typedef ap_int<B_BIT_WIDTH> bias_t;


// The top-level function
void predict(
        imap_t input_values[IHEIGHT * IWIDTH * ICHANNELS],
        weigth_t weights[IC2D_1_KSIZE * IC2D_1_KSIZE],
        bias_t bias[IC2D_1_BSIZE],
        omap_t output[IC2D_1_IHEIGHT * IC2D_1_IWIDTH * IC2D_1_ICHANNELS]
    );
/*
void conv2D_1(
        imap_t img_in[IC2D_1_IHEIGHT * IC2D_1_IWIDTH * IC2D_1_ICHANNELS],
        weigth_t weights[IC2D_1_KSIZE * IC2D_1_KSIZE],
		bias_t bias[IC2D_1_BSIZE],
		omap_t img_out[IC2D_1_OHEIGHT * IC2D_1_OWIDTH * IC2D_1_OCHANNELS]
    );
*/

int main() {
	int i, j, c, err_cnt = 0;

	printf("Input Image\n");
	for (i = 0; i < IHEIGHT; ++i) {
		for (j = 0; j < IWIDTH; ++j) {
			//img_in[i * IWIDTH + j] = (i + 1) * 10 + (j + 1);
			printf("%2d ", img_in[i * IWIDTH + j]);
		}
		printf("\n");
	}

	predict(
			(imap_t*) img_in,
            (weigth_t*) kernel,
            (bias_t*) bias,
			(omap_t*) img_out
	    );
	/*
	conv2D_1(
        (imap_t*) img_in,
        (weigth_t*) kernel,
        (bias_t*) bias,
        (omap_t*) img_out
    );
    */

	printf("\nOutput Image\n");
	for (c = 0; c < IC2D_1_ICHANNELS; ++c) {
        const int channel_offset = c * IC2D_1_IWIDTH * IC2D_1_IHEIGHT;
		printf("CHANNEL %d\n", c);
		for (i = 0; i < IC2D_1_IHEIGHT; ++i) {
			for (j = 0; j < IC2D_1_IWIDTH; ++j) {
				printf("%2d ", img_out[(i * IC2D_1_IWIDTH + j) + channel_offset]);
			}
			printf("\n");
		}
		printf("\n");
	}


	/*
	printf("\nOutput Image\n");
	for (i = 0; i < OHEIGHT; i++) {
		for (j = 0; j < OWIDTH; j++) {
			printf("%4d ", img_out[i * OWIDTH + j]);
		}
		printf("\n");
	}

    /*
	for (err_cnt = 0, i = 0; i < OHEIGHT; i++) {
		for (j = 0; j < OWIDTH; j++) {
			if (hw_img_out[i * OWIDTH + j] != sw_img_out[i * OWIDTH + j]) {
				err_cnt++;
				printf("%d,%d: %d != %d\n", i, j, hw_img_out[i * OWIDTH + j],
						sw_img_out[i * OWIDTH + j]);
			}
		}
	}
	
    return err_cnt;
    */

    return 0;
}
