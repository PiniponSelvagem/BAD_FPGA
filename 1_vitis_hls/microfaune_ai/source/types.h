#ifndef MY_TYPES
#define MY_TYPES

#include "global_settings.h"

/* GLOBAL TYPES */
typedef ap_int<W_BIT_WIDTH*PACKET> weigth_t;
typedef ap_int<I_BIT_WIDTH*PACKET> imap_t;
typedef ap_uint<I_BIT_WIDTH*PACKET> omap_t;

/* LAYER TYPES */
#define W W_BIT_WIDTH
#define I 1

typedef ap_fixed<W,I, AP_RND, AP_SAT> quant_t;


#endif // !MY_TYPES
