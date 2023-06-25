#include "input.h"

void input_preconv2d(const input_t input[INPUT_LINES][INPUT_COLS], conv_t inputPad[INPUT_PAD_LINES][INPUT_PAD_COLS]) {
    /*
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
    */

    for (int w = 0; w < (INPUT_PAD_COLS - 1); ++w) {
        inputPad[0][w] = 0;
    }

    for (int hP = 1, h = 0; hP < (INPUT_PAD_LINES - 1); ++hP, ++h) {
        inputPad[hP][0] = 0;
        for (int wP = 1, w = 0; wP < (INPUT_PAD_COLS - 1); ++wP, ++w) {
            inputPad[hP][wP] = input[h][w];
        }
        inputPad[h][(INPUT_PAD_COLS - 1)] = 0;
    }

    for (int w = 0; w < (INPUT_PAD_COLS - 1); ++w) {
        inputPad[(INPUT_PAD_LINES - 1)][w] = 0;
    }
}
