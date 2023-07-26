#pragma once

#ifndef MY_TYPES
#define MY_TYPES

//#define TRUNCATE_BITS

/* GLOBAL TYPES */
#ifdef __VITIS_HLS__
#include <ap_int.h>
#include <ap_fixed.h>
// margin included
typedef ap_uint<7> i64_t;
typedef ap_uint<8> i128_t;
typedef ap_int<10> i512_t;  // requires signal because backward layer has check: i >= 0, and i will be -1
#define TC(x) x
#endif
#ifdef _MSC_VER
typedef int i64_t;
typedef int i128_t;
typedef int i512_t;
#define TRUNC_MAX_VALUE(bits) ((1 << (bits)) - 1)
#define TRUNC_X(x, bits) ((int)((x) * TRUNC_MAX_VALUE(bits)) / (float)TRUNC_MAX_VALUE(bits))
#ifndef TRUNCATE_BITS
#define TC(x) x
#else
// 10 bits, seems to be the absolute minimum
// 12 bits, safe bet
#define TC(x) TRUNC_X(x, 12)
#endif // TRUNCATE_BITS
#endif

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

#define BNORM_EPSILON   0.001
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

#define BNORM_EPSILON   0.001
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
