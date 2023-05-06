#pragma once
#include "data_tensorflow.h"

void test_sig_z_values();
void test_sig_r_values();
void test_tanh_hh_values();

void gru_tensorflow(gruval* input, gruval* state, gruval* output);