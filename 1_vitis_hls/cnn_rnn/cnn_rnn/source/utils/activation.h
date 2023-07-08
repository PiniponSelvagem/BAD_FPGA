#pragma once

#ifndef ACTIVATION_H
#define ACXTIVATION


#include "sigmoid.h"
//#include <math.h>
#include "tanh_rnnoise.h"
#define SIGMOID(x)          sigmoid(x)
#define TANH(x)             tansig_approx(x) //tanh(x)


#endif // ACTIVATION_H