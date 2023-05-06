#pragma once

#ifndef SETTINGS_H
#define SETTINGS_H


#define FLOAT
#ifdef FLOAT
    typedef float gruval;
#else
    typedef double gruval;
#endif


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