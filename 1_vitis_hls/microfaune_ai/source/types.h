#pragma once

#ifndef MY_TYPES
#define MY_TYPES

//#define TRUNCATE_BITS     //TRUNCATE_BITS TRUNCATE_BITS TRUNCATE_BITS TRUNCATE_BITS TRUNCATE_BITS

/* GLOBAL TYPES */
#include <ap_int.h>
#include <ap_fixed.h>
// margin included
typedef ap_uint<7> i64_t;
typedef ap_uint<8> i128_t;
typedef ap_int<10> i512_t;  // requires signal because backward layer has check: i >= 0, and i will be -1
#define TC(x) x


/* LAYER TYPES */
#define W 4
#define I 1
#define Wacc W+5
#define Iacc I+2

typedef ap_fixed<W,I, AP_RND, AP_SAT> input_t;
typedef ap_fixed<W,I, AP_RND, AP_SAT> input_t;
typedef ap_fixed<W,I, AP_RND, AP_SAT> conv_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> conv_acc_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> mpool_t;
typedef ap_fixed<W,I, AP_RND, AP_SAT> reducemax_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> gru_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> gru_acc_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> timedist_t;
typedef ap_fixed<Wacc,Iacc, AP_RND, AP_SAT> timedist_acc_t;

typedef ap_fixed<W,I, AP_RND, AP_SAT> output_t;

typedef float sigmoid_t;
typedef float tanh_t;


#endif // !MY_TYPES
