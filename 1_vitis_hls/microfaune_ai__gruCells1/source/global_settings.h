#ifndef GLOBAL_SETTINGS
#define GLOBAL_SETTINGS

//#define DEBUG_MODEL


#define BUS_WIDTH 64
#define W_BIT_WIDTH 4			// 16
#define I_BIT_WIDTH 4			// 16 but TensorFlow uses float/double, so maybe 32 or 64?
#define BIAS_SIZE 16

#define PACKET_CNN   16			// number of weights inside each transfer of axis in CNN, conv
#define PACKET_GRU   8			// number of weights inside each transfer of axis in GRU

#ifdef DEBUG_MODEL
#define CHANNELS    16
#else
#define CHANNELS	64			// cannot be lower than PACKET
#endif
#define FILTERS     CHANNELS


#endif // !GLOBAL_SETTINGS
