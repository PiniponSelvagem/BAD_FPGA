#pragma once

#ifndef GRU_SETTINGS_H
#define GRU_SETTINGS_H

#include "../global_settings.h"
#include "../types.h"

#define GRU_STATE_SIZE      64

#include "utils/sigmoid.h"
#include <math.h>
#define SIGMOID(x)          sigmoid(x);
#define TANH(x)             tanh(x);        // WARNING: MATH.H is using DOUBLE


#endif // GRU_SETTINGS_H