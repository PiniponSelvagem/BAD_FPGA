#ifndef SIZE_BGRU
#define SIZE_BGRU

#include "size_conv3D.h"

#define GRU_FORWARD  1
#define GRU_BACKWARD 0

#define GRU_SPLIT_SIZE    3

#define GRU_FILTERS		1
#define GRU_IN_LINES	IHEIGHT
#define GRU_IN_COLS		CHANNELS

#define GRU_BIAS_SIZE       (GRU_FILTERS*GRU_SPLIT_SIZE)
#define GRU_KERNEL_COLS_MAX	FILTERS		// x2 to act has a limiter, related to GRU_1 backward kernel size
#define GRU_KERNEL_REC_COLS	GRU_FILTERS


#define GRU_INCOLS_MAX      GRU_KERNEL_COLS_MAX
#define GRU_0__IN_COLS		GRU_IN_COLS
#define GRU_0__KERNEL_COLS	GRU_FILTERS

#endif // !SIZE_BGRU
