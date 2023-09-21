#pragma once

#ifndef TANH_Q4_H
#define TANH_Q4_H

#include "../types.h"

#define OFFSET      0.0625
#define STEP_SIZE   0.125

static inline tanh_t tanh_table(tanh_t input) {
    input = input + OFFSET;
    if (input <= -1)
        return -1.0;
    else if (input >= 0.875)
        return 0.875;
    int step = int((input + 1) / STEP_SIZE);
    return (step * STEP_SIZE) - 1.0;
}

/*
#define TQ4_OFFSET 0.06
static const tanh_t tanh_q4_table[16] = {
    -1.000000,
    -0.875000,
    -0.750000,
    -0.625000,
    -0.500000,
    -0.375000,
    -0.250000,
    -0.125000,
     0.000000,
     0.125000,
     0.250000,
     0.375000,
     0.500000,
     0.625000,
     0.750000,
     0.875000,
};
static const int tanh_q4_table_size = sizeof(tanh_q4_table) / sizeof(tanh_q4_table[0]);



static inline tanh_t tanh_table(tanh_t input) {
    input = (input < -1.0) ? -1.0 : ((input > 1.0) ? 1.0 : input);      // in range [-1, 1]
    int index = (int)((input + 1.0) * 8.0);

    if (index > (tanh_q4_table_size - 1))
        index = tanh_q4_table_size - 1;
    else if (index < 0)
        index = 0;

    return tanh_q4_table[index];
}
*/

#endif