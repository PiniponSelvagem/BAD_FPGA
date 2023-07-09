#pragma once

#ifndef BNORM_5_H
#define BNORM_5_H

#include "../../global_settings.h"
#include "../bnorm_settings.h"
#include "../../conv2d/data/conv2d_5.h"

#define BNORM_5__RAW_IN_LINES     C2D_5__RAW_IN_LINES
#define BNORM_5__RAW_IN_COLS      C2D_5__RAW_IN_COLS
#define BNORM_5__RAW_OUT_LINES    BNORM_5__RAW_IN_LINES
#define BNORM_5__RAW_OUT_COLS     BNORM_5__RAW_IN_COLS

#define BNORM_5__IN_LINES         (BNORM_5__RAW_IN_LINES + PADDING)
#define BNORM_5__IN_COLS          (BNORM_5__RAW_IN_COLS + PADDING)

#define BNORM_5__OUT_LINES        BNORM_5__IN_LINES
#define BNORM_5__OUT_COLS         BNORM_5__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t gamma_5[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t beta_5[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t movingmean_5[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t movingvariance_5[CHANNELS];


#endif // BNORM_5_H