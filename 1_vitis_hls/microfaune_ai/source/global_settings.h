#ifndef GLOBAL_SETTINGS
#define GLOBAL_SETTINGS


#define W_BIT_WIDTH 4          // 16
#define I_BIT_WIDTH 4          // 16 but TensorFlow uses float/double, so maybe 32 or 64?

#define PACKET      16         // number of weights inside each tranfer of axis

#define CHANNELS    16          // 64
#define FILTERS     CHANNELS


#endif // !GLOBAL_SETTINGS
