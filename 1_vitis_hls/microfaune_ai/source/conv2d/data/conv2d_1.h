#pragma once

#ifndef CONV2D_1_H
#define CONV2D_1_H

#include "../../global_settings.h"
#include "../conv2d_settings.h"
#include "../../bnorm/data/bnorm_0.h"

#define C2D_1__RAW_IN_LINES     BNORM_0__RAW_IN_LINES
#define C2D_1__RAW_IN_COLS      BNORM_0__RAW_IN_COLS
#define C2D_1__RAW_OUT_LINES    C2D_1__RAW_IN_LINES
#define C2D_1__RAW_OUT_COLS     C2D_1__RAW_IN_COLS

#define C2D_1__IN_LINES         (C2D_1__RAW_IN_LINES + PADDING)
#define C2D_1__IN_COLS          (C2D_1__RAW_IN_COLS + PADDING)

#define C2D_1__OUT_LINES        C2D_1__IN_LINES
#define C2D_1__OUT_COLS         C2D_1__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
conv_t kernel_1[CHANNELS][FILTERS][C2D_KERNEL_LINES][C2D_KERNEL_COLS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
conv_t bias_1[C2D_BIAS_SIZE];

#endif // CONV2D_1_H