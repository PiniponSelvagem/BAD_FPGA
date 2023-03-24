#include <ap_int.h>
#include "data.h"
#include "../Source/settings.h"


typedef ap_int<W_BIT_WIDTH> weigth_t;
typedef ap_uint<I_BIT_WIDTH> imap_t;
typedef ap_uint<I_BIT_WIDTH> omap_t;
typedef ap_int<B_BIT_WIDTH> bias_t;


// The top-level function
void predict(
        imap_t input_values[IHEIGHT * IWIDTH * ICHANNELS],
        weigth_t weights[IC2D_1_KSIZE * IC2D_1_KSIZE * IC2D_1_ICHANNELS],
        bias_t bias[IC2D_1_BSIZE],
        omap_t output[IC2D_1_OHEIGHT * IC2D_1_OWIDTH * IC2D_1_OCHANNELS]
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
	for (c = 0; c < IC2D_1_OCHANNELS; ++c) {
        const int channel_offset = c * IC2D_1_OWIDTH * IC2D_1_OHEIGHT;
		printf("CHANNEL %d\n", c);
		for (i = 0; i < IC2D_1_OHEIGHT; ++i) {
			for (j = 0; j < IC2D_1_OWIDTH; ++j) {
				printf("%2d ", img_out[(i * IC2D_1_OWIDTH + j) + channel_offset]);
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
