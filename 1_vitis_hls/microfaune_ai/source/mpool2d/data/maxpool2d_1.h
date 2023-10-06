#pragma once

#ifndef MP2D_1_H
#define MP2D_1_H

#include "../../global_settings.h"
#include "../../conv2d/data/conv2d_3.h"

#define MP2D_1__RAW_IN_LINES     C2D_3__RAW_OUT_LINES
#define MP2D_1__RAW_IN_COLS      C2D_3__RAW_OUT_COLS
#define MP2D_1__RAW_OUT_LINES    MP2D_1__RAW_IN_LINES
#define MP2D_1__RAW_OUT_COLS     (MP2D_1__RAW_IN_COLS/2)

#define MP2D_1__IN_LINES         (MP2D_1__RAW_IN_LINES + PADDING)
#define MP2D_1__IN_COLS          (MP2D_1__RAW_IN_COLS + PADDING)

#define MP2D_1__OUT_LINES        (MP2D_1__RAW_OUT_LINES + PADDING)
#define MP2D_1__OUT_COLS         (MP2D_1__RAW_OUT_COLS + PADDING)

#endif // MPV2D_1_H