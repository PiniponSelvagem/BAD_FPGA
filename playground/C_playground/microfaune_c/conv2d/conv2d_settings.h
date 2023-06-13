#pragma once
#include "../types.h"

#ifndef CONV2D_SETTINGS_H
#define CONV2D_SETTINGS_H


#define C2D_CHANNELS         64

#define C2D_KERNEL_LINES     3
#define C2D_KERNEL_COLS      3

#define C2D_BIAS_SIZE        C2D_CHANNELS

#define C2D_PADDING          2
#define C2D_OFFSET           (C2D_PADDING/2)


#endif // CONV2D_SETTINGS_H