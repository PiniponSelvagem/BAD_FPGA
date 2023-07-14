#pragma once

#ifndef OUTPUT_H
#define OUTPUT_H

#include "../global_settings.h"
#include "../types.h"
#include "../timedist/data/timedist_1.h"
#include "../reducemax/data/reducemax_1.h"

#define OUTPUT_LOCAL_SCORE_LINES    INPUT_LINES
#define OUTPUT_LOCAL_SCORE_COLS     TD_1__OUT_COLS
#define OUTPUT_GLOBAL_SCORE         RMAX_1__OUT_COLS

#endif // OUTPUT_H