#pragma once

#ifndef MY_TYPES
#define MY_TYPES

#include <ap_int.h>

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
#else
typedef ap_fixed<16,1> input_t;
typedef ap_fixed<16,6> conv_t;
typedef ap_fixed<16,6> bnorm_t;
typedef ap_fixed<16,6> mpool_t;
typedef ap_fixed<16,6> reducemax_t;
typedef ap_fixed<16,1> gru_t;
typedef ap_fixed<16,1> timedist_t;
typedef ap_fixed<16,1> output_t;
#endif

#endif // !MY_TYPES
