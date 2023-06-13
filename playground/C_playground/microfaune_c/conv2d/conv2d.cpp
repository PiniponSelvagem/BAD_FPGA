
#include "conv2d.h"

void conv2d_preprocess(const conv_t input[INPUT_LINES][INPUT_COLS], conv_t inputPad[C2D_0__IN_LINES][C2D_0__IN_COLS]) {
    for (int h = 0; h < (C2D_0__IN_LINES - 1); ++h) {
        for (int w = 0; w < (C2D_0__IN_COLS - 1); ++w) {
            if (h == 0 || h == (C2D_0__IN_LINES - 1) || w == 0 || w == (C2D_0__IN_COLS - 1)) {
                inputPad[h][w] = 0;
            }
            else {
                inputPad[h][w] = input[h - C2D_OFFSET][w - C2D_OFFSET];
            }
        }
    }
}

/*
template <int C_IN_LINES, int C_IN_COLS, int C_OUT_LINES, int C_OUT_COLS>
void conv2d(
    const conv_t input[C_IN_LINES][C_IN_COLS],
    const conv_t weights[C2D_KERNEL_LINES][C2D_KERNEL_COLS],
    const conv_t bias,
    conv_t output[C_OUT_LINES][C_OUT_COLS]
) {
    for (int orow = 1; orow < (C_OUT_LINES - C2D_OFFSET); ++orow) {
        for (int ocol = C2D_OFFSET; ocol < (C_OUT_COLS - C2D_OFFSET); ++ocol) {
            conv_t acc = bias;
            conv_t acc_sat;
            for (int krow = 0; krow < C2D_KERNEL_LINES; ++krow) {
                for (int kcol = 0; kcol < C2D_KERNEL_COLS; ++kcol) {
                    acc += weights[krow][kcol] * input[orow + krow - C2D_OFFSET][ocol + kcol - C2D_OFFSET];
                }
            }

            /*
            if (acc > 255)
                acc_sat = 255;
            else if (acc < 0)
                acc_sat = 0;    // ReLu
            else
            *
            acc_sat = acc;
            output[orow][ocol] = acc_sat;
        }
    }
}
*/

void conv2d_postprocess(const conv_t prepro[C2D_0__IN_LINES][C2D_0__IN_COLS], conv_t output[INPUT_LINES][INPUT_COLS]) {
    for (int h = 0; h < (C2D_0__IN_LINES - 1); ++h) {
        for (int w = 0; w < (C2D_0__IN_COLS - 1); ++w) {
            if (h == 0 || h == (C2D_0__IN_LINES - 1) || w == 0 || w == (C2D_0__IN_COLS - 1)) {
                ;
            }
            else {
                output[h - C2D_OFFSET][w - C2D_OFFSET] = prepro[h][w];
            }
        }
    }
}