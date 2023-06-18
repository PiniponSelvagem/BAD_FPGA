#pragma once

#ifndef MP2D_0_H
#define MP2D_0_H

#include "../../global_settings.h"
#include "../../bnorm/data/bnorm_1.h"

#define MP2D_0__RAW_IN_LINES     BNORM_1__RAW_OUT_LINES
#define MP2D_0__RAW_IN_COLS      BNORM_1__RAW_OUT_COLS
#define MP2D_0__RAW_OUT_LINES    MP2D_0__RAW_IN_LINES
#define MP2D_0__RAW_OUT_COLS     (MP2D_0__RAW_IN_COLS/2)

#define MP2D_0__IN_LINES         (MP2D_0__RAW_IN_LINES + PADDING)
#define MP2D_0__IN_COLS          (MP2D_0__RAW_IN_COLS + PADDING)

#define MP2D_0__OUT_LINES        (MP2D_0__RAW_OUT_LINES + PADDING)
#define MP2D_0__OUT_COLS         (MP2D_0__RAW_OUT_COLS + PADDING)

#endif // MPV2D_0_H