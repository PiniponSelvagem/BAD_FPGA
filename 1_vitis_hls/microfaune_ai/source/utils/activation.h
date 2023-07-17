#pragma once

#ifndef ACTIVATION_H
#define ACXTIVATION


#include "sigmoid.h"
#include <math.h>
//#include "tanh_rnnoise.h"
#define SIGMOID(x)          sigmoid((float)(x))
#define TANH(x)             tanh((float)(x)) //tansig_approx(x)    


#endif // ACTIVATION_H