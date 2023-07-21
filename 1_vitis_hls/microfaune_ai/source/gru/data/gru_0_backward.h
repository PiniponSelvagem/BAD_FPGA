#pragma once

#ifndef GRU_0_B_H
#define GRU_0_B_H

#include "gru_0.h"

/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t kernel_gru0_b[GRU_0__KERNEL_LINES][GRU_SPLIT][GRU_0__KERNEL_COLS];
/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t recurrent_kernel_gru0_b[GRU_0__KERNEL_R_LINES][GRU_SPLIT][GRU_0__KERNEL_R_COLS];

/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t bias_gru0_b[GRU_0__KERNEL_COLS][GRU_SPLIT];
/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t recurrent_bias_gru0_b[GRU_0__KERNEL_COLS][GRU_SPLIT];

#endif // GRU_0_B_H