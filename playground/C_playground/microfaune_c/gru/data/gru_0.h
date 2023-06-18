#pragma once

#ifndef GRU_0_H
#define GRU_0_H

#include "../gru_settings.h"
#include "../../reducemax/data/reducemax_0.h"

#define GRU_0__IN_LINES         RMAX_0__OUT_COLS
#define GRU_0__IN_COLS          RMAX_0__OUT_LINES
#define GRU_0__OUT_LINES        GRU_0__IN_LINES
#define GRU_0__OUT_COLS         (GRU_0__IN_COLS*2)

#define GRU_0__KERNEL_LINES     GRU_0__IN_COLS
#define GRU_0__KERNEL_COLS      (GRU_0__IN_COLS*3)

#define GRU_0__KERNEL_R_LINES   GRU_0__KERNEL_LINES
#define GRU_0__KERNEL_R_COLS    GRU_0__KERNEL_COLS

#define GRU_0__BIAS_SIZE        GRU_0__KERNEL_COLS
#define GRU_0__SPLIT_SIZE       (GRU_0__KERNEL_COLS/3)


#endif // GRU_0_H