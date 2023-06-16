#include "input.h"

void input_preconv2d(const input_t input[INPUT_LINES][INPUT_COLS], conv_t inputPad[INPUT_PAD_LINES][INPUT_PAD_COLS]) {
    for (int h = 0; h < (INPUT_PAD_LINES - 1); ++h) {
        for (int w = 0; w < (INPUT_PAD_COLS - 1); ++w) {
            if (h == 0 || h == (INPUT_PAD_LINES - 1) || w == 0 || w == (INPUT_PAD_COLS - 1)) {
                inputPad[h][w] = 0;
            }
            else {
                inputPad[h][w] = input[h - PADDING_OFFSET][w - PADDING_OFFSET];
            }
        }
    }
}
