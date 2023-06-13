#pragma once

#ifndef SIGMOID_H
#define SIGMOID_H

#include <math.h>

// Taken from: https://github.com/nayayayay/sigmoid-function

#define EULER_NUMBER 2.71828
#define EULER_NUMBER_F 2.71828182846
#define EULER_NUMBER_L 2.71828182845904523536

float inline sigmoid(float n) {
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
}

double inline sigmoidd(double n) {
    return (1 / (1 + pow(EULER_NUMBER, -n)));
}

long double inline sigmoidl(long double n) {
    return (1 / (1 + powl(EULER_NUMBER_L, -n)));
}

#endif