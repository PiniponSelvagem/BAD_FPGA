#include "conv2d.h"

#include <stdio.h>
#include <string.h>



void conv2d_preprocess(const convval input[IHEIGHT][IWIDTH], convval inputPad[C2D_1_IHEIGHT][C2D_1_IWIDTH]) {
    for (int h = 0; h < (C2D_1_IHEIGHT-1); ++h) {
        for (int w = 0; w < (C2D_1_IWIDTH-1); ++w) {
            if (h == 0 || h == (C2D_1_IHEIGHT - 1) || w == 0 || w == (C2D_1_IWIDTH - 1)) {
                inputPad[h][w] = 0;
            }
            else {
                inputPad[h][w] = input[h-C2D_OFFSET][w-C2D_OFFSET];
            }
        }
    }
}

void conv2d(
    const convval input[C2D_1_IHEIGHT][C2D_1_IWIDTH],
    const convval weights[C2D_1_KSIZE][C2D_1_KSIZE],
    const convval bias,
    convval output[C2D_1_OHEIGHT][C2D_1_OWIDTH])
{
    for (int orow = 1; orow < (C2D_1_OHEIGHT - C2D_OFFSET); ++orow) {
        for (int ocol = C2D_OFFSET; ocol < (C2D_1_OWIDTH - C2D_OFFSET); ++ocol) {
            convval acc = bias;
            convval acc_sat;
            for (int krow = 0; krow < C2D_1_KSIZE; ++krow) {
                for (int kcol = 0; kcol < C2D_1_KSIZE; ++kcol) {
                    acc += weights[krow][kcol] * input[orow + krow - C2D_OFFSET][ocol + kcol - C2D_OFFSET];
                }
            }

            /*
            if (acc > 255)
                acc_sat = 255;
            else if (acc < 0)
                acc_sat = 0;    // ReLu
            else
            */
                acc_sat = acc;
            output[orow][ocol] = acc_sat;
        }
    }
}

void conv2d_postprocess(const convval prepro[C2D_1_IHEIGHT][C2D_1_IWIDTH], convval output[IHEIGHT][IWIDTH]) {
    for (int h = 0; h < (C2D_1_IHEIGHT - 1); ++h) {
        for (int w = 0; w < (C2D_1_IWIDTH - 1); ++w) {
            if (h == 0 || h == (C2D_1_IHEIGHT - 1) || w == 0 || w == (C2D_1_IWIDTH - 1)) {
                ;
            }
            else {
                output[h - C2D_OFFSET][w - C2D_OFFSET] = prepro[h][w];
            }
        }
    }
}