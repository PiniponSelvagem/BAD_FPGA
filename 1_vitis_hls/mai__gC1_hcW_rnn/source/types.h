#ifndef MY_TYPES
#define MY_TYPES

#include "global_settings.h"
#include <ap_int.h> 
#include <ap_fixed.h>

/* GLOBAL TYPES */
/* CONV */
typedef ap_int<W_BIT_WIDTH*PACKET_CNN> weigth_t;
typedef ap_int<BIAS_SIZE> bias_t;
typedef ap_uint<I_BIT_WIDTH*PACKET_CNN> imap_t;		// 2023-11-24 before was ap_int
typedef ap_uint<I_BIT_WIDTH*PACKET_CNN> omap_t;

/* GRU */
typedef ap_fixed<G_WG_W_BIT_WIDTH, G_WG_I_BIT_WIDTH, AP_RND, AP_SAT> gru_weigth_t;
/* GRU internal types */
#define I_MXB 4     // max bits used on average for matrix_x array
typedef ap_fixed<G_IN_W_BIT_WIDTH,G_IN_I_BIT_WIDTH, AP_RND, AP_SAT> gru_imap_t;
typedef ap_fixed<I_MXB+G_IN_W_BIT_WIDTH+G_WG_W_BIT_WIDTH, I_MXB, AP_RND, AP_SAT> gru_matrix_t;
typedef ap_ufixed<G_S_W_BIT_WIDTH,G_S_I_BIT_WIDTH, AP_RND, AP_SAT> gru_sigmoid_t;
typedef ap_fixed<G_T_W_BIT_WIDTH,G_T_I_BIT_WIDTH, AP_RND, AP_SAT> gru_tanh_t;
typedef ap_fixed<G_IN_W_BIT_WIDTH,G_IN_I_BIT_WIDTH, AP_RND, AP_SAT> gru_omap_t;

/* LAYER TYPES */
#define W W_BIT_WIDTH
#define I 1

typedef ap_fixed<W,I, AP_RND, AP_SAT> quant_t;

#endif // !MY_TYPES
