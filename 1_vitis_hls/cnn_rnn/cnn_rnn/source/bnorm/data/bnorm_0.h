#pragma once

#ifndef BNORM_0_H
#define BNORM_0_H

#include "../../global_settings.h"
#include "../bnorm_settings.h"
#include "../../input/input.h"
#include "../../conv2d/data/conv2d_0.h"

#define BNORM_0__RAW_IN_LINES     C2D_0__RAW_IN_LINES
#define BNORM_0__RAW_IN_COLS      C2D_0__RAW_IN_COLS
#define BNORM_0__RAW_OUT_LINES    BNORM_0__RAW_IN_LINES
#define BNORM_0__RAW_OUT_COLS     BNORM_0__RAW_IN_COLS

#define BNORM_0__IN_LINES         (BNORM_0__RAW_IN_LINES + PADDING)
#define BNORM_0__IN_COLS          (BNORM_0__RAW_IN_COLS + PADDING)

#define BNORM_0__OUT_LINES        BNORM_0__IN_LINES
#define BNORM_0__OUT_COLS         BNORM_0__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t gamma_0[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t beta_0[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t movingmean_0[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t movingvariance_0[CHANNELS];


#endif // BNORM_0_H