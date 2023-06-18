#pragma once

#ifndef GRU_1_H
#define GRU_1_H

#include "../gru_settings.h"
#include "gru_0.h"

#define GRU_1__IN_LINES         GRU_0__OUT_LINES
#define GRU_1__IN_COLS          GRU_0__OUT_COLS
#define GRU_1__OUT_LINES        GRU_1__IN_LINES
#define GRU_1__OUT_COLS         GRU_1__IN_COLS          // mantains the output size of 128

#define GRU_1__KERNEL_LINES     GRU_1__IN_COLS
#define GRU_1__KERNEL_COLS      ((GRU_1__IN_COLS/2)*3)

#define GRU_1__KERNEL_R_LINES   (GRU_1__KERNEL_LINES/2)
#define GRU_1__KERNEL_R_COLS    GRU_1__KERNEL_COLS

#define GRU_1__BIAS_SIZE        GRU_1__KERNEL_COLS
#define GRU_1__SPLIT_SIZE       (GRU_1__KERNEL_COLS/3)


#endif // GRU_1_H