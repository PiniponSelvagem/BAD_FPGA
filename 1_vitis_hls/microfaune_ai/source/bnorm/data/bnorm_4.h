#pragma once

#ifndef BNORM_4_H
#define BNORM_4_H

#include "../../global_settings.h"
#include "../bnorm_settings.h"
#include "../../conv2d/data/conv2d_4.h"

#define BNORM_4__RAW_IN_LINES     C2D_4__RAW_IN_LINES
#define BNORM_4__RAW_IN_COLS      C2D_4__RAW_IN_COLS
#define BNORM_4__RAW_OUT_LINES    BNORM_4__RAW_IN_LINES
#define BNORM_4__RAW_OUT_COLS     BNORM_4__RAW_IN_COLS

#define BNORM_4__IN_LINES         (BNORM_4__RAW_IN_LINES + PADDING)
#define BNORM_4__IN_COLS          (BNORM_4__RAW_IN_COLS + PADDING)

#define BNORM_4__OUT_LINES        BNORM_4__IN_LINES
#define BNORM_4__OUT_COLS         BNORM_4__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t gamma_4[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t beta_4[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t movingmean_4[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t movingvariance_4[CHANNELS];


#endif // BNORM_4_H