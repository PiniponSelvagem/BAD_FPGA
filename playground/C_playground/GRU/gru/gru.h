#pragma once
#include "../data/data_static.h"

void gru_clearState();
void gru_syncState();
void gru(int idx, const gruval* input, gruval* output);