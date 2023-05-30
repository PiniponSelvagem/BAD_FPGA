#pragma once

#ifndef GRU_SETTINGS_H
#define GRU_SETTINGS_H

#include "../utils/types.h"

#define INPUT_SIZE  64
#define KERNEL_ROWS INPUT_SIZE
#define KERNEL_COLS (INPUT_SIZE*3)
#define BIAS_COLS   KERNEL_COLS
#define SPLIT_SIZE  (KERNEL_COLS/3)
#define OUTPUT_SIZE INPUT_SIZE

   //#define TANH_RRNOISE
#ifdef TANH_RRNOISE
    #include "../utils/sigmoid.h"
    #include "../utils/tanh_rnnoise.h"
    #ifdef FLOAT
        #define SIGMOID(x)  sigmoid(x);
        #define TANH(x)     tanh(x);
    #else
        #define SIGMOID(x)  sigmoidd(x);
        #define TANH(x)     tanh(x);
    #endif
#else
    #include "../utils/sigmoid.h"
    #include <math.h>
    #ifdef FLOAT
        #define SIGMOID(x)  sigmoid(x);
        #define TANH(x)     tanh(x);        // WARNING: MATH.H is using DOUBLE
    #else
        #define SIGMOID(x)  sigmoidd(x);
        #define TANH(x)     tanh(x);
    #endif
#endif

#endif