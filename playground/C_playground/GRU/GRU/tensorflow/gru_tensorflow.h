#pragma once
#include "../data/data_static.h"

void test_sig_z_values();
void test_sig_r_values();
void test_tanh_hh_values();

void gru_tf_clearState();

void gru_cell(int idx, const gruval* input, gruval* state, gruval* output);
void gru_tensorflow(const gruval* input, gruval* output);