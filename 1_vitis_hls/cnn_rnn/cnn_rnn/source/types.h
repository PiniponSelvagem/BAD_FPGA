#pragma once

#ifndef MY_TYPES
#define MY_TYPES

#include <ap_int.h>
#include <ap_fixed.h>

#define USE_FLOAT
#ifdef USE_FLOAT
typedef float input_t;
typedef float conv_t;
typedef float bnorm_t;
typedef float mpool_t;
typedef float reducemax_t;
typedef float gru_t;
typedef float timedist_t;
typedef float output_t;

typedef float sigmoid_t;
typedef float tanh_t;

#else
typedef ap_fixed<16,7> input_t;
typedef ap_fixed<16,7> input_t;
typedef ap_fixed<16,7> conv_t;
typedef ap_fixed<16,7> bnorm_t;
typedef ap_fixed<16,7> mpool_t;
typedef ap_fixed<16,7> reducemax_t;
typedef ap_fixed<16,7> gru_t;
typedef ap_fixed<16,7> timedist_t;
typedef ap_fixed<16,7> output_t;

typedef float sigmoid_t;
typedef float tanh_t;
#endif

#endif // !MY_TYPES
