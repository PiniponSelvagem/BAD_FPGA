#pragma once

#ifndef TIMEDIST__0_H
#define TIMEDIST__0_H

#include "../../global_settings.h"
#include "../timedist_settings.h"
#include "../../gru/data/gru_1.h"

#define TD_0__RAW_IN_LINES     1
#define TD_0__RAW_IN_COLS      GRU_1__OUT_COLS
#define TD_0__RAW_OUT_LINES    TD_0__RAW_IN_LINES
#define TD_0__RAW_OUT_COLS     TD_0__RAW_IN_COLS

#define TD_0__IN_LINES         TD_0__RAW_IN_LINES
#define TD_0__IN_COLS          TD_0__RAW_IN_COLS

#define TD_0__OUT_LINES        TD_0__IN_LINES
#define TD_0__OUT_COLS         (TD_0__IN_COLS/2)

#define TD_0__KERNEL_LINES     GRU_1__OUT_COLS
#define TD_0__KERNEL_COLS      CHANNELS

#define TD_0__BIAS_SIZE        CHANNELS

/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
timedist_t kernel_td0[TD_0__KERNEL_LINES][TD_0__KERNEL_COLS];

/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
timedist_t bias_td0[TD_0__BIAS_SIZE];


#endif // TIMEDIST__0_H