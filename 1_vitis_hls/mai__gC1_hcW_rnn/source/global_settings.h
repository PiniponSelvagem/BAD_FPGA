#ifndef GLOBAL_SETTINGS
#define GLOBAL_SETTINGS

//#define DEBUG_MODEL


#define BUS_WIDTH 64

/* CONV */
#define W_BIT_WIDTH 4			// 16
#define I_BIT_WIDTH 4			// 16 but TensorFlow uses float/double, so maybe 32 or 64?
#define BIAS_SIZE 16

/* GRU */
#define G_IN_W_BIT_WIDTH 8
#define G_IN_I_BIT_WIDTH 1
#define G_WG_W_BIT_WIDTH 8
#define G_WG_I_BIT_WIDTH 1
#define G_S_W_BIT_WIDTH 8
#define G_S_I_BIT_WIDTH 0
#define G_T_W_BIT_WIDTH 8
#define G_T_I_BIT_WIDTH 1
#define G_STATE_W_BIT_WIDTH (G_IN_W_BIT_WIDTH*2)
#define G_STATE_I_BIT_WIDTH G_IN_I_BIT_WIDTH

#define PACKET_CNN   16			// number of weights inside each transfer of axis in CNN, conv
#define PACKET_GRU   8			// number of weights inside each transfer of axis in GRU

#ifdef DEBUG_MODEL
#define CHANNELS    16
#else
#define CHANNELS	64			// cannot be lower than PACKET
#endif
#define FILTERS     CHANNELS


#endif // !GLOBAL_SETTINGS