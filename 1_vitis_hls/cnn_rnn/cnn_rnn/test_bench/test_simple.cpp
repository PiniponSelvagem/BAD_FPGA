#include <ap_int.h>
#include "../Source/settings.h"
#include "data.h"


void predict(
        imap_t input_values[IHEIGHT * IWIDTH * ICHANNELS],
        weigth_t weights[C2D_1_KSIZE * C2D_1_KSIZE * C2D_1_ICHANNELS],
        bias_t bias[C2D_1_BSIZE],
        omap_t output[C2D_1_OHEIGHT * C2D_1_OWIDTH * C2D_1_OCHANNELS]
    );

void conv2D_1(
        imap_t img_in[C2D_1_IHEIGHT * C2D_1_IWIDTH * C2D_1_ICHANNELS],
        weigth_t weights[C2D_1_KSIZE * C2D_1_KSIZE],
		bias_t bias[C2D_1_BSIZE],
		omap_t img_out[C2D_1_OHEIGHT * C2D_1_OWIDTH * C2D_1_OCHANNELS]
    );

// The top-level function
void maxpooling2D_1(
        imap_t input[MP2D_1_IHEIGHT * MP2D_1_IWIDTH * MP2D_1_CHANNELS],
        omap_t output[MP2D_1_OHEIGHT * MP2D_1_OWIDTH * MP2D_1_CHANNELS]
    );



void test_predict() {
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

    printf("\nOutput Image\n");
	for (c = 0; c < C2D_1_OCHANNELS; ++c) {
        const int channel_offset = c * C2D_1_OWIDTH * C2D_1_OHEIGHT;
		printf("CHANNEL %d\n", c);
		for (i = 0; i < C2D_1_OHEIGHT; ++i) {
			for (j = 0; j < C2D_1_OWIDTH; ++j) {
				printf("%2d ", img_out[(i * C2D_1_OWIDTH + j) + channel_offset]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

static unsigned char s_imgC_in[C2D_1_IHEIGHT * C2D_1_IWIDTH * C2D_1_ICHANNELS];
void test_conv2D() {
    int i, j, c, err_cnt = 0;
    printf("Input Image\n");
	for (i = 0; i < IHEIGHT; ++i) {
		for (j = 0; j < IWIDTH; ++j) {
			//img_in[i * IWIDTH + j] = (i + 1) * 10 + (j + 1);
			printf("%2d ", img_in[i * IWIDTH + j]);
		}
		printf("\n");
	}

    // prepare
    for (int channel = 0; channel < C2D_1_ICHANNELS; ++channel) {
        const int channel_offset = channel * C2D_1_IWIDTH * C2D_1_IHEIGHT;
        for (int orow = 0; orow < C2D_1_IHEIGHT; ++orow) {
            for (int ocol = 0; ocol < C2D_1_IWIDTH; ++ocol) {
                const int index = (orow * C2D_1_IWIDTH + ocol) + channel_offset;
                if (orow < C2D_OFFSET || orow >= C2D_OFFSET+IHEIGHT ||
                    ocol < C2D_OFFSET || ocol >= C2D_OFFSET+IWIDTH) {
                    s_imgC_in[index] = 0;
                }
                else {
                    char value = img_in[(orow-C2D_OFFSET) * IWIDTH + (ocol-C2D_OFFSET)];
                    s_imgC_in[index] = value;
                    //printf("c %2d -> %2d -> %d\n", channel, input[index], index);
                }
            }
        }
    }
    
	conv2D_1(
        (imap_t*) s_imgC_in,
        (weigth_t*) kernel,
        (bias_t*) bias,
        (omap_t*) img_out
    );

    printf("\nOutput Image\n");
	for (c = 0; c < C2D_1_OCHANNELS; ++c) {
        const int channel_offset = c * C2D_1_OWIDTH * C2D_1_OHEIGHT;
		printf("CHANNEL %d\n", c);
		for (i = 0; i < C2D_1_OHEIGHT; ++i) {
			for (j = 0; j < C2D_1_OWIDTH; ++j) {
				printf("%2d ", img_out[(i * C2D_1_OWIDTH + j) + channel_offset]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

void test_maxpooling2D() {
    int i, j, c, err_cnt = 0;
    printf("Input Image\n");
	for (i = 0; i < IHEIGHT; ++i) {
		for (j = 0; j < IWIDTH; ++j) {
			printf("%2d ", maxp_in[i * IWIDTH + j]);
		}
		printf("\n");
	}
    
	maxpooling2D_1(
        (imap_t*) maxp_in,
        (omap_t*) maxp_out
    );

    printf("\nOutput Image\n");
	for (c = 0; c < MP2D_1_CHANNELS; ++c) {
        const int channel_offset = c * MP2D_1_OWIDTH * MP2D_1_OHEIGHT;
		printf("CHANNEL %d\n", c);
		for (i = 0; i < MP2D_1_OHEIGHT; ++i) {
			for (j = 0; j < MP2D_1_OWIDTH; ++j) {
				printf("%2d ", maxp_out[(i * MP2D_1_OWIDTH + j) + channel_offset]);
			}
			printf("\n");
		}
		printf("\n");
	}
}