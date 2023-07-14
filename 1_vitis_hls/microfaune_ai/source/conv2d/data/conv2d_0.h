#pragma once

#ifndef CONV2D_0_H
#define CONV2D_0_H

#include "../../global_settings.h"
#include "../conv2d_settings.h"
#include "../../input/input.h"

#define C2D_0__RAW_IN_LINES     INPUT_LINES
#define C2D_0__RAW_IN_COLS      INPUT_COLS
#define C2D_0__RAW_OUT_LINES    C2D_0__RAW_IN_LINES
#define C2D_0__RAW_OUT_COLS     C2D_0__RAW_IN_COLS

#define C2D_0__IN_LINES         (C2D_0__RAW_IN_LINES + PADDING)
#define C2D_0__IN_COLS          (C2D_0__RAW_IN_COLS + PADDING)

#define C2D_0__OUT_LINES        C2D_0__IN_LINES
#define C2D_0__OUT_COLS         C2D_0__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
conv_t kernel_0[CHANNELS][C2D_KERNEL_LINES][C2D_KERNEL_COLS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
conv_t bias_0[C2D_BIAS_SIZE];


#endif // CONV2D_0_H