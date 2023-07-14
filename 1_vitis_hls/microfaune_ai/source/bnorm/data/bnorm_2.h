#pragma once

#ifndef BNORM_2_H
#define BNORM_2_H

#include "../../global_settings.h"
#include "../bnorm_settings.h"
#include "../../conv2d/data/conv2d_2.h"

#define BNORM_2__RAW_IN_LINES     C2D_2__RAW_IN_LINES
#define BNORM_2__RAW_IN_COLS      C2D_2__RAW_IN_COLS
#define BNORM_2__RAW_OUT_LINES    BNORM_2__RAW_IN_LINES
#define BNORM_2__RAW_OUT_COLS     BNORM_2__RAW_IN_COLS

#define BNORM_2__IN_LINES         (BNORM_2__RAW_IN_LINES + PADDING)
#define BNORM_2__IN_COLS          (BNORM_2__RAW_IN_COLS + PADDING)

#define BNORM_2__OUT_LINES        BNORM_2__IN_LINES
#define BNORM_2__OUT_COLS         BNORM_2__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t gamma_2[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t beta_2[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t movingmean_2[CHANNELS];

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
bnorm_t movingvariance_2[CHANNELS];


#endif // BNORM_2_H