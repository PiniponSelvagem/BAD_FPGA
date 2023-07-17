#pragma once

#ifndef MY_TYPES
#define MY_TYPES

#include <ap_int.h>
#include <ap_fixed.h>

/* GLOBAL TYPES */
// margin included
typedef ap_uint<7> i64_t;
typedef ap_uint<8> i128_t;
typedef ap_int<10> i512_t;  // requires signal because backward layer has check: i >= 0, and i will be -1

/* LAYER TYPES */
#define USE_FLOAT
//#define USE_16_7
#ifdef USE_FLOAT
typedef float input_t;
typedef float conv_t;

#define BNORM_EPSILON   0.001
typedef float bnorm_t;

typedef float mpool_t;
typedef float reducemax_t;
typedef float gru_t;
typedef float timedist_t;
typedef float output_t;

typedef float sigmoid_t;
typedef float tanh_t;

#else
#ifdef USE_16_7
typedef ap_fixed<16,7> input_t;
typedef ap_fixed<16,7> input_t;
typedef ap_fixed<16,7> conv_t;

#define BNORM_EPSILON   (1.0/2/2/2/2/2/2/2/2/2)     // i know this is ridiculous, but atm i have no idea of a simplier way (the number of 2s is equal to the number of decimal bits)
typedef ap_fixed<16,7> bnorm_t;

typedef ap_fixed<16,7> mpool_t;
typedef ap_fixed<16,7> reducemax_t;
typedef ap_fixed<16,7> gru_t;
typedef ap_fixed<16,7> timedist_t;
typedef ap_fixed<16,7> output_t;

typedef float sigmoid_t;
typedef float tanh_t;
#else
typedef ap_fixed<32,8> input_t;
typedef ap_fixed<32,8> input_t;
typedef ap_fixed<32,8> conv_t;

#define BNORM_EPSILON   (1.0/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2)
typedef ap_fixed<32,8> bnorm_t;

typedef ap_fixed<32,8> mpool_t;
typedef ap_fixed<32,8> reducemax_t;
typedef ap_fixed<32,8> gru_t;
typedef ap_fixed<32,8> timedist_t;
typedef ap_fixed<32,8> output_t;

typedef float sigmoid_t;
typedef float tanh_t;
#endif
#endif

#endif // !MY_TYPES
