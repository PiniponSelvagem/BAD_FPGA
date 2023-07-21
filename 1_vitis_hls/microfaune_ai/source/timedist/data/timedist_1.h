#pragma once

#ifndef TIMEDIST__1_H
#define TIMEDIST__1_H

#include "../../global_settings.h"
#include "../timedist_settings.h"
#include "timedist_0.h"

#define TD_1__RAW_IN_LINES     1
#define TD_1__RAW_IN_COLS      TD_0__OUT_COLS
#define TD_1__RAW_OUT_LINES    TD_1__RAW_IN_LINES
#define TD_1__RAW_OUT_COLS     TD_1__RAW_IN_COLS

#define TD_1__IN_LINES         TD_1__RAW_IN_LINES
#define TD_1__IN_COLS          TD_1__RAW_IN_COLS

#define TD_1__OUT_LINES        TD_1__IN_LINES
#define TD_1__OUT_COLS         1

#define TD_1__KERNEL_LINES     1
#define TD_1__KERNEL_COLS      CHANNELS

#define TD_1__BIAS_SIZE        1

/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
timedist_t kernel_td1[TD_1__KERNEL_LINES][TD_1__KERNEL_COLS];

/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
timedist_t bias_td1[TD_1__BIAS_SIZE];


#endif // TIMEDIST__1_H