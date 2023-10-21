#pragma once

#ifndef GRU_0_F_H
#define GRU_0_F_H

#include "gru_0.h"

/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t kernel_gru0_f[GRU_0__KERNEL_LINES][GRU_0__KERNEL_COLS][GRU_SPLIT];
gru_t kernel_gru0_f_scale[GRU_0__KERNEL_LINES][GRU_SPLIT];
/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t recurrent_kernel_gru0_f[GRU_0__KERNEL_R_LINES][GRU_0__KERNEL_R_COLS][GRU_SPLIT];
gru_t recurrent_kernel_gru0_f_scale[GRU_0__KERNEL_R_LINES][GRU_SPLIT];

/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t bias_gru0_f[GRU_0__KERNEL_COLS][GRU_SPLIT];

#endif // GRU_0_F_H