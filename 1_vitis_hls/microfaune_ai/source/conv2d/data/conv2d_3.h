#pragma once

#ifndef CONV2D_3_H
#define CONV2D_3_H

#include "../../global_settings.h"
#include "../conv2d_settings.h"
#include "conv2d_2.h"

#define C2D_3__RAW_IN_LINES     C2D_2__RAW_OUT_LINES
#define C2D_3__RAW_IN_COLS      C2D_2__RAW_OUT_COLS
#define C2D_3__RAW_OUT_LINES    C2D_3__RAW_IN_LINES
#define C2D_3__RAW_OUT_COLS     C2D_3__RAW_IN_COLS

#define C2D_3__IN_LINES         (C2D_3__RAW_IN_LINES + PADDING)
#define C2D_3__IN_COLS          (C2D_3__RAW_IN_COLS + PADDING)

#define C2D_3__OUT_LINES        C2D_3__IN_LINES
#define C2D_3__OUT_COLS         C2D_3__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
conv_t kernel_3[FILTERS][CHANNELS][C2D_KERNEL_LINES][C2D_KERNEL_COLS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
conv_t bias_3[C2D_BIAS_SIZE];


conv_t kernel_3_scale[FILTERS];


#endif // CONV2D_3_H