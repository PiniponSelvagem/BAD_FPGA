#include "../../source/global_settings.h"
#include "../../source/input/input_settings.h"

void soft_input_preconv2d(const input_t input[INPUT_LINES][INPUT_COLS], conv_t inputPad[INPUT_PAD_LINES][INPUT_PAD_COLS]) {

    INPUT_loop_Lstart: for (int w = 0; w < (INPUT_PAD_COLS - 1); ++w) {
        inputPad[0][w] = 0;
    }

	INPUT_loop_H: for (int hP = 1, h = 0; hP < (INPUT_PAD_LINES - 1); ++hP, ++h) {
        inputPad[hP][0] = 0;
    	INPUT_loop_W: for (int wP = 1, w = 0; wP < (INPUT_PAD_COLS - 1); ++wP, ++w) {
            inputPad[hP][wP] = input[h][w];
        }
        inputPad[hP][(INPUT_PAD_COLS - 1)] = 0;
    }

    INPUT_loop_Lend: for (int w = 0; w < (INPUT_PAD_COLS - 1); ++w) {
        inputPad[(INPUT_PAD_LINES - 1)][w] = 0;
    }
}