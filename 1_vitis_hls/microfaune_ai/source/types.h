#pragma once

#ifndef MY_TYPES
#define MY_TYPES

//#define TRUNCATE_BITS

/* GLOBAL TYPES */
#ifdef __VITIS_HLS__
#include <ap_int.h>
#include <ap_fixed.h>
//#define USE_FLOAT
#define USE_8_1
//#define USE_16_7
// margin included
typedef ap_uint<7> i64_t;
typedef ap_uint<8> i128_t;
typedef ap_int<10> i512_t;  // requires signal because backward layer has check: i >= 0, and i will be -1
#define TC(x) x
#endif
#ifdef _MSC_VER
#define USE_FLOAT
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
#ifdef USE_FLOAT
typedef float input_t;
typedef float conv_t;
typedef float conv_acc_t;

#define BNORM_EPSILON   0.001
typedef float bnorm_t;
typedef float bnorm_acc_t;

typedef float mpool_t;
typedef float reducemax_t;

typedef float gru_t;
typedef float gru_acc_t;

typedef float timedist_t;
typedef float timedist_acc_t;

typedef float output_t;

typedef float sigmoid_t;
typedef float tanh_t;

#else
#ifdef USE_8_1
#define W 8
#define I 1
#define Wacc W+5
#define Iacc I+2

typedef ap_fixed<W,I, AP_RND, AP_SAT> input_t;
typedef ap_fixed<W,I, AP_RND, AP_SAT> input_t;
typedef ap_fixed<W,I, AP_RND, AP_SAT> conv_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> conv_acc_t;

#define BNORM_EPSILON   0.125
typedef ap_fixed<W,I, AP_RND, AP_SAT> bnorm_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> bnorm_acc_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> mpool_t;
typedef ap_fixed<W,I, AP_RND, AP_SAT> reducemax_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> gru_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> gru_acc_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> timedist_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> timedist_acc_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> output_t;

typedef float sigmoid_t;
typedef float tanh_t;
#else
#ifdef USE_16_7
#define W 16
#define I 7
#define Wacc W+5
#define Iacc I+2

typedef ap_fixed<W,I, AP_RND, AP_SAT> input_t;
typedef ap_fixed<W,I, AP_RND, AP_SAT> input_t;
typedef ap_fixed<W,I, AP_RND, AP_SAT> conv_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> conv_acc_t;

#define BNORM_EPSILON   0.01
typedef ap_fixed<W,I, AP_RND, AP_SAT> bnorm_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> bnorm_acc_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> mpool_t;
typedef ap_fixed<W,I, AP_RND, AP_SAT> reducemax_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> gru_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> gru_acc_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> timedist_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> timedist_acc_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> output_t;

typedef float sigmoid_t;
typedef float tanh_t;
#else
#define W 30 //30 //32 //32 //32
#define I 6  //6  //6  //7  //8
#define Wacc W+5
#define Iacc I+2

typedef ap_fixed<W,I, AP_RND, AP_SAT> input_t;
typedef ap_fixed<W,I, AP_RND, AP_SAT> input_t;
typedef ap_fixed<W,I, AP_RND, AP_SAT> conv_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> conv_acc_t;

#define BNORM_EPSILON   0.001
typedef ap_fixed<W,I, AP_RND, AP_SAT> bnorm_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> bnorm_acc_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> mpool_t;
typedef ap_fixed<W,I, AP_RND, AP_SAT> reducemax_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> gru_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> gru_acc_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> timedist_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> timedist_acc_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> output_t;

typedef float sigmoid_t;
typedef float tanh_t;
#endif
#endif
#endif

#ifndef USE_FLOAT
#define MAX_VALUE ((1 << (W)) - 1)
#endif

#endif // !MY_TYPES
