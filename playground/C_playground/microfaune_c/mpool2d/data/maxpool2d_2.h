#pragma once

#ifndef MP2D_2_H
#define MP2D_2_H

#include "../../global_settings.h"
#include "../../bnorm/data/bnorm_5.h"

#define MP2D_2__RAW_IN_LINES     BNORM_5__RAW_OUT_LINES
#define MP2D_2__RAW_IN_COLS      BNORM_5__RAW_OUT_COLS
#define MP2D_2__RAW_OUT_LINES    MP2D_2__RAW_IN_LINES
#define MP2D_2__RAW_OUT_COLS     (MP2D_2__RAW_IN_COLS/2)

#define MP2D_2__IN_LINES         (MP2D_2__RAW_IN_LINES + PADDING)
#define MP2D_2__IN_COLS          (MP2D_2__RAW_IN_COLS + PADDING)

#define MP2D_2__OUT_LINES        (MP2D_2__RAW_OUT_LINES + PADDING)
#define MP2D_2__OUT_COLS         (MP2D_2__RAW_OUT_COLS + PADDING)

#endif // MPV2D_2_H