#pragma once

#ifndef BNORM_1_H
#define BNORM_1_H

#include "../../global_settings.h"
#include "../bnorm_settings.h"
#include "../../conv2d/data/conv2d_1.h"

#define BNORM_1__RAW_IN_LINES     C2D_1__RAW_IN_LINES
#define BNORM_1__RAW_IN_COLS      C2D_1__RAW_IN_COLS
#define BNORM_1__RAW_OUT_LINES    BNORM_1__RAW_IN_LINES
#define BNORM_1__RAW_OUT_COLS     BNORM_1__RAW_IN_COLS

#define BNORM_1__IN_LINES         (BNORM_1__RAW_IN_LINES + PADDING)
#define BNORM_1__IN_COLS          (BNORM_1__RAW_IN_COLS + PADDING)

#define BNORM_1__OUT_LINES        BNORM_1__IN_LINES
#define BNORM_1__OUT_COLS         BNORM_1__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t gamma_1[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t beta_1[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t movingmean_1[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t movingvariance_1[CHANNELS];


#endif // BNORM_1_H