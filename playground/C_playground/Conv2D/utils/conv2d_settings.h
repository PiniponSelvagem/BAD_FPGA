#pragma once

#ifndef CONV_SETTINGS_H
#define CONV_SETTINGS_H

#include "../utils/types.h"

#define INPUT_SIZE          40
#define INPUT_SIZE_OUTTER   431
#define OUTPUT_SIZE         INPUT_SIZE
#define OUTPUT_SIZE_OUTTER  INPUT_SIZE_OUTTER


#define PADDING_SAME 2
#define C2D_OFFSET   (PADDING_SAME/2)

/* InputLayer */
#define IHEIGHT   3
#define IWIDTH    40
#define ICHANNELS 1


#define CNN_CHANNELS    64

/* Conv2D_1 */
#define C2D_1_IHEIGHT   (IHEIGHT + PADDING_SAME)
#define C2D_1_IWIDTH    (IWIDTH + PADDING_SAME)
#define C2D_1_ICHANNELS CNN_CHANNELS

#define C2D_1_KSIZE     3
#define C2D_1_BSIZE	    CNN_CHANNELS

#define C2D_1_OHEIGHT   C2D_1_IHEIGHT
#define C2D_1_OWIDTH    C2D_1_IWIDTH
#define C2D_1_OCHANNELS C2D_1_ICHANNELS

#endif

