#pragma once

#ifndef GRU_1_B_H
#define GRU_1_B_H

#include "gru_1.h"

/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t kernel_gru1_b[GRU_1__KERNEL_LINES][GRU_1__KERNEL_COLS][GRU_SPLIT];
/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t recurrent_kernel_gru1_b[GRU_1__KERNEL_LINES][GRU_1__KERNEL_COLS][GRU_SPLIT];

/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t bias_gru1_b[GRU_1__KERNEL_COLS][GRU_SPLIT];
/**
 * @brief This was taken from "model_json__channel_last/dump_weights_bias.json". (Channels last)
*/
gru_t recurrent_bias_gru1_b[GRU_1__KERNEL_COLS][GRU_SPLIT];

#endif // GRU_1_F_H