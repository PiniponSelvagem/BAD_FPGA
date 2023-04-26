#pragma once


#ifdef OPTION_DEBUG
float input[2] = { 10, 11 };

#elif OPTION_1
float input[2] = { 1, 1 };

#elif OPTION_2
float input[2] = { -2, 1 };

#elif OPTION_3
float input[4] = { 0.01, 0.02, 0.03, 0.04 };

#else
float input[2] = { 1, 1 };

#endif