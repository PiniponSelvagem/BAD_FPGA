#ifndef GLOBAL_SETTINGS
#define GLOBAL_SETTINGS

#define DEBUG_MODEL


#define W_BIT_WIDTH 4			// 16
#define I_BIT_WIDTH 4			// 16 but TensorFlow uses float/double, so maybe 32 or 64?

#define PACKET      16			// number of weights inside each transfer of axis

#ifdef DEBUG_MODEL
#define CHANNELS    16
#else
#define CHANNELS	64			// cannot be lower than PACKET
#endif
#define FILTERS     CHANNELS


#endif // !GLOBAL_SETTINGS
