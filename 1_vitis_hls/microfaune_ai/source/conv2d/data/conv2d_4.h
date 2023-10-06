#pragma once

#ifndef CONV2D_4_H
#define CONV2D_4_H

#include "../../global_settings.h"
#include "../conv2d_settings.h"
#include "../../mpool2d/data/maxpool2d_1.h"

#define C2D_4__RAW_IN_LINES     MP2D_1__RAW_OUT_LINES
#define C2D_4__RAW_IN_COLS      MP2D_1__RAW_OUT_COLS
#define C2D_4__RAW_OUT_LINES    C2D_4__RAW_IN_LINES
#define C2D_4__RAW_OUT_COLS     C2D_4__RAW_IN_COLS

#define C2D_4__IN_LINES         (C2D_4__RAW_IN_LINES + PADDING)
#define C2D_4__IN_COLS          (C2D_4__RAW_IN_COLS + PADDING)

#define C2D_4__OUT_LINES        C2D_4__IN_LINES
#define C2D_4__OUT_COLS         C2D_4__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
conv_t kernel_4[FILTERS][CHANNELS][C2D_KERNEL_LINES][C2D_KERNEL_COLS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
conv_t bias_4[C2D_BIAS_SIZE];


conv_t kernel_4_scale[FILTERS];


#endif // CONV2D_4_H