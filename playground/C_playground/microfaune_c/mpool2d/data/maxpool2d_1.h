#pragma once
#include "../../global_settings.h"
#include "../../bnorm/data/bnorm_3.h"

#ifndef MP2D_1_H
#define MP2D_1_H

#define MP2D_1__RAW_IN_LINES     BNORM_3__RAW_OUT_LINES
#define MP2D_1__RAW_IN_COLS      BNORM_3__RAW_OUT_COLS
#define MP2D_1__RAW_OUT_LINES    MP2D_1__RAW_IN_LINES
#define MP2D_1__RAW_OUT_COLS     (MP2D_1__RAW_IN_COLS/2)

#define MP2D_1__IN_LINES         (MP2D_1__RAW_IN_LINES + PADDING)
#define MP2D_1__IN_COLS          (MP2D_1__RAW_IN_COLS + PADDING)

#define MP2D_1__OUT_LINES        (MP2D_1__RAW_OUT_LINES + PADDING)
#define MP2D_1__OUT_COLS         (MP2D_1__RAW_OUT_COLS + PADDING)

#endif // MPV2D_0_H