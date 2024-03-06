#ifndef GLOBAL_SETTINGS
#define GLOBAL_SETTINGS

#define IHEIGHT		431
#define IWIDTH		40

/* CONV */
#define W_BIT_WIDTH 4			// 16
#define I_BIT_WIDTH 4			// 16 but TensorFlow uses float/double, so maybe 32 or 64?

/* GRU */
#define GRU_IN_W_BIT_WIDTH 8
#define GRU_IN_I_BIT_WIDTH 1

#define GRU_CELLS	1

#define CHANNELS	64
#define FILTERS     CHANNELS


#endif // !GLOBAL_SETTINGS
