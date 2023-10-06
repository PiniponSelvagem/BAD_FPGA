#pragma once

#ifndef CONV2D_5_H
#define CONV2D_5_H

#include "../../global_settings.h"
#include "../conv2d_settings.h"
#include "conv2d_4.h"

#define C2D_5__RAW_IN_LINES     C2D_4__RAW_OUT_LINES
#define C2D_5__RAW_IN_COLS      C2D_4__RAW_OUT_COLS
#define C2D_5__RAW_OUT_LINES    C2D_5__RAW_IN_LINES
#define C2D_5__RAW_OUT_COLS     C2D_5__RAW_IN_COLS

#define C2D_5__IN_LINES         (C2D_5__RAW_IN_LINES + PADDING)
#define C2D_5__IN_COLS          (C2D_5__RAW_IN_COLS + PADDING)

#define C2D_5__OUT_LINES        C2D_5__IN_LINES
#define C2D_5__OUT_COLS         C2D_5__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
conv_t kernel_5[FILTERS][CHANNELS][C2D_KERNEL_LINES][C2D_KERNEL_COLS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
conv_t bias_5[C2D_BIAS_SIZE];


conv_t kernel_5_scale[FILTERS];


#endif // CONV2D_5_H