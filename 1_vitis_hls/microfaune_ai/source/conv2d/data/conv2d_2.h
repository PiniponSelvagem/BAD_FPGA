#pragma once

#ifndef CONV2D_2_H
#define CONV2D_2_H

#include "../../global_settings.h"
#include "../conv2d_settings.h"
#include "../../mpool2d/data/maxpool2d_0.h"

#define C2D_2__RAW_IN_LINES     MP2D_0__RAW_OUT_LINES
#define C2D_2__RAW_IN_COLS      MP2D_0__RAW_OUT_COLS
#define C2D_2__RAW_OUT_LINES    C2D_2__RAW_IN_LINES
#define C2D_2__RAW_OUT_COLS     C2D_2__RAW_IN_COLS

#define C2D_2__IN_LINES         (C2D_2__RAW_IN_LINES + PADDING)
#define C2D_2__IN_COLS          (C2D_2__RAW_IN_COLS + PADDING)

#define C2D_2__OUT_LINES        C2D_2__IN_LINES
#define C2D_2__OUT_COLS         C2D_2__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
conv_t kernel_2[FILTERS][CHANNELS][C2D_KERNEL_LINES][C2D_KERNEL_COLS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
conv_t bias_2[C2D_BIAS_SIZE];


#endif // CONV2D_2_H