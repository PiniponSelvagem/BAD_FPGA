#pragma once

#ifndef MY_TYPES
#define MY_TYPES

#define FLOAT
#ifdef FLOAT
    typedef float convval;
#else
    typedef double convval;
#endif

#endif // !MY_TYPES
