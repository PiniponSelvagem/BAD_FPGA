#pragma once

#ifndef GLOBAL_SETTINGS_H
#define GLOBAL_SETTINGS_H

#include "types.h"

#define CHANNELS        64
#define FILTERS         64

#define PADDING         2
#define PADDING_OFFSET  (PADDING/2)
#define CNN_LINES       431
#define CNN_COLS        40
#define CNN_LINES_PAD   (CNN_LINES+PADDING)
#define CNN_COLS_PAD    (CNN_COLS+PADDING)


#endif // GLOBAL_SETTINGS_H