#include "timedist.h"


/*
void conv2d_postprocess(const conv_t prepro[C2D_0__IN_LINES][C2D_0__IN_COLS], conv_t output[INPUT_LINES][INPUT_COLS]) {
    for (int h = 0; h < (C2D_0__IN_LINES - 1); ++h) {
        for (int w = 0; w < (C2D_0__IN_COLS - 1); ++w) {
            if (h == 0 || h == (C2D_0__IN_LINES - 1) || w == 0 || w == (C2D_0__IN_COLS - 1)) {
                ;
            }
            else {
                output[h - PADDING_OFFSET][w - PADDING_OFFSET] = prepro[h][w];
            }
        }
    }
}
*/