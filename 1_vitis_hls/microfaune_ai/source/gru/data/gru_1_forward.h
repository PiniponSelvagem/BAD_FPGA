#pragma once

#ifndef GRU_1_F_H
#define GRU_1_F_H

#include "gru_1.h"

/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t kernel_gru1_f[GRU_1__KERNEL_LINES][GRU_1__KERNEL_COLS][GRU_SPLIT];
gru_t kernel_gru1_f_scale[GRU_1__KERNEL_LINES][GRU_SPLIT];
/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t recurrent_kernel_gru1_f[GRU_1__KERNEL_R_LINES][GRU_1__KERNEL_R_COLS][GRU_SPLIT];
gru_t recurrent_kernel_gru1_f_scale[GRU_1__KERNEL_R_LINES][GRU_SPLIT];

/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t bias_gru1_f[GRU_1__KERNEL_COLS][GRU_SPLIT];

#endif // GRU_1_F_H