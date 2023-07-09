#pragma once

#ifndef BNORM_3_H
#define BNORM_3_H

#include "../../global_settings.h"
#include "../bnorm_settings.h"
#include "../../conv2d/data/conv2d_3.h"

#define BNORM_3__RAW_IN_LINES     C2D_3__RAW_IN_LINES
#define BNORM_3__RAW_IN_COLS      C2D_3__RAW_IN_COLS
#define BNORM_3__RAW_OUT_LINES    BNORM_3__RAW_IN_LINES
#define BNORM_3__RAW_OUT_COLS     BNORM_3__RAW_IN_COLS

#define BNORM_3__IN_LINES         (BNORM_3__RAW_IN_LINES + PADDING)
#define BNORM_3__IN_COLS          (BNORM_3__RAW_IN_COLS + PADDING)

#define BNORM_3__OUT_LINES        BNORM_3__IN_LINES
#define BNORM_3__OUT_COLS         BNORM_3__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t gamma_3[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t beta_3[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t movingmean_3[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t movingvariance_3[CHANNELS];


#endif // BNORM_3_H