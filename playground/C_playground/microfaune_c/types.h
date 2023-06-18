#pragma once

#ifndef MY_TYPES
#define MY_TYPES

#define FLOAT
#ifdef FLOAT
    typedef float input_t;
    typedef float conv_t;
    typedef float bnorm_t;
    typedef float mpool_t;
    typedef float reducemax_t;
    typedef float gru_t;

    typedef float output_t;
#else
    #error "Not implemented"
#endif

#endif // !MY_TYPES
