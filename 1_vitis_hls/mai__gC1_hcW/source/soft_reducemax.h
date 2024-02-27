#ifndef REDUCEMAX_H
#define REDUCEMAX_H

#include "global_settings.h"
#include "size_bgru.h"

#define RMAX_MIN_VALUE  0

void reducemax_1(
    float* input,
    float* output
) {
	float maxval = RMAX_MIN_VALUE;
    RMAX_1_loop_row: for (int row = 0; row < IHEIGHT; ++row) {
    	float value = *(input + row);
        if (value > maxval) {
            maxval = value;
        }
    }
    *output = maxval;
}


#endif // REDUCEMAX_H
