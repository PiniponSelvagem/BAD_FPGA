#pragma once
#include "../data/data_static.h"

void conv2d_preprocess(const convval input[IHEIGHT][IWIDTH], convval inputPad[C2D_1_IHEIGHT][C2D_1_IWIDTH]);
void conv2d(const convval input[C2D_1_IHEIGHT][C2D_1_IWIDTH], const convval weights[C2D_1_KSIZE][C2D_1_KSIZE], const convval bias, convval output[C2D_1_OHEIGHT][C2D_1_OWIDTH]);
void conv2d_postprocess(const convval prepro[C2D_1_IHEIGHT][C2D_1_IWIDTH], convval output[IHEIGHT][IWIDTH]);
