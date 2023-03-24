#define HW_IP 1
#include "../Source/image_sizes.h"

static unsigned char img_in[IHEIGHT * IWIDTH];
static signed char kernel[K_SIZE * K_SIZE] = { -2, -1, 0, -1, 1, 1, 0, 1, 2 };
static int bias = 5;

static unsigned char hw_img_out[OHEIGHT * OWIDTH];
static unsigned char sw_img_out[OHEIGHT * OWIDTH];

void sw_convolution_2D(unsigned char *img_in, unsigned char *img_out) {
	signed char *weights = kernel;
	int acc, weight_1d_loc, image_1d_loc;
	int orow, ocol, krow, kcol;

	for (orow = 0; orow < OHEIGHT; orow++) {
		for (ocol = 0; ocol < OWIDTH; ocol++) {
			acc = bias;
			for (krow = 0; krow < K_SIZE; krow++) {
				for (kcol = 0; kcol < K_SIZE; kcol++) {
					weight_1d_loc = (krow) * K_SIZE + (kcol);
					image_1d_loc = (orow + krow) * IWIDTH + (ocol + kcol);
					acc += weights[weight_1d_loc] * img_in[image_1d_loc];
					//				  printf("%d += %d * %d\n", acc, weights[weight_1d_loc], img_in[image_1d_loc]);
				}
			}
			if (acc > 255)
				acc = 255;
			else if (acc < 0)
				acc = 0;
			img_out[orow * OWIDTH + ocol] = acc;
		}
	}
}

#include <ap_int.h>
typedef ap_int<8> weigth_t;
typedef ap_uint<8> imap_t;
typedef ap_uint<8> omap_t;
typedef ap_int<32> bias_t;

// The top-level function
void axil_conv2D0(imap_t img_in[IHEIGHT * IWIDTH],
		omap_t img_out[OHEIGHT * OWIDTH], weigth_t weights[K_SIZE * K_SIZE],
		bias_t bias);

int main() {
	int i, j, err_cnt = 0;

	printf("Input Image\n");
	for (i = 0; i < IHEIGHT; i++) {
		for (j = 0; j < IWIDTH; j++) {
			img_in[i * IWIDTH + j] = (i + 1) * 10 + (j + 1);
			printf("%d ", img_in[i * IWIDTH + j]);
		}
		printf("\n");
	}

#if HW_IP
	axil_conv2D0((imap_t*) img_in, (omap_t*) hw_img_out, (weigth_t*) kernel,
			(bias_t) bias);
#endif

	sw_convolution_2D(img_in, sw_img_out);

	printf("\nOutput Image\n");
	for (i = 0; i < OHEIGHT; i++) {
		for (j = 0; j < OWIDTH; j++) {
			printf("%4d ", sw_img_out[i * OWIDTH + j]);
		}
		printf("\n");
	}

#if HW_IP
	for (err_cnt = 0, i = 0; i < OHEIGHT; i++) {
		for (j = 0; j < OWIDTH; j++) {
			if (hw_img_out[i * OWIDTH + j] != sw_img_out[i * OWIDTH + j]) {
				err_cnt++;
				printf("%d,%d: %d != %d\n", i, j, hw_img_out[i * OWIDTH + j],
						sw_img_out[i * OWIDTH + j]);
			}
		}
	}
#endif

	return err_cnt;
}
