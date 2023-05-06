#pragma once
#include "settings.h"

#define INPUT_SIZE  4
#define KERNEL_ROWS INPUT_SIZE
#define KERNEL_COLS (INPUT_SIZE*3)
#define BIAS_COLS   KERNEL_COLS
#define SPLIT_SIZE  (KERNEL_COLS/3)
#define OUTPUT_SIZE INPUT_SIZE

// dump_weights_bias.json (channel last)

const gruval kernel[KERNEL_ROWS][KERNEL_COLS] = {
    {
        0.09328797459602356,
        -0.14423318207263947,
        0.05723040923476219,
        -0.11457829177379608,
        -0.1551823765039444,
        0.06890249997377396,
        0.14696961641311646,
        0.15685993432998657,
        0.08041565865278244,
        -0.011380927637219429,
        0.11137595772743225,
        0.039657384157180786,
    },
    {
        -0.06359504163265228,
        0.010740764439105988,
        -0.06599920243024826,
        0.0035890135914087296,
        -0.06796855479478836,
        -0.014085154980421066,
        0.0003849963250104338,
        0.2191721647977829,
        -0.14060650765895844,
        0.13892300426959991,
        -0.07230409979820251,
        -0.05099346861243248,
    },
    {
        0.054023921489715576,
        0.029494358226656914,
        -0.13789314031600952,
        -0.0066886041313409805,
        -0.10313959419727325,
        -0.06582074612379074,
        0.09140714257955551,
        -0.10954931378364563,
        -0.01481624785810709,
        0.07574597001075745,
        0.0783919021487236,
        0.0569973848760128,
    },
    {
        0.10220101475715637,
        -0.0894460529088974,
        -0.192433699965477,
        0.15619510412216187,
        -0.13574598729610443,
        -0.015100651420652866,
        -0.04062913358211517,
        -0.05308453366160393,
        -0.03741860389709473,
        0.1041494831442833,
        -0.10299824178218842,
        0.10386911779642105,
    },
};
const gruval recurrent_kernel[KERNEL_ROWS][KERNEL_COLS] = {
    {
        0.02673882246017456,
        0.057039160281419754,
        0.014511843211948872,
        -0.09925466030836105,
        -0.046919357031583786,
        -0.005756886675953865,
        -0.08231764286756516,
        0.04614228755235672,
        0.04263906925916672,
        0.028190460056066513,
        0.04845970869064331,
        -0.0447242446243763,
    },
    {
        -0.048061519861221313,
        -0.13918046653270721,
        0.0632314532995224,
        0.008528322912752628,
        0.15494024753570557,
        0.06547942012548447,
        0.08842337131500244,
        0.0030862062703818083,
        -0.027148408815264702,
        -0.12858068943023682,
        0.02790527418255806,
        -0.14804261922836304,
    },
    {
        0.0975215807557106,
        0.05268855392932892,
        0.11988946050405502,
        0.03758050501346588,
        0.12552648782730103,
        0.05336257070302963,
        0.047844476997852325,
        0.040932852774858475,
        -0.0773303359746933,
        0.054036278277635574,
        0.17089463770389557,
        0.037518974393606186,
    },
    {
        -0.03923599049448967,
        0.09807618707418442,
        -0.1178690567612648,
        0.1491660326719284,
        0.003437004517763853,
        -0.028483103960752487,
        0.11781293898820877,
        0.06325460225343704,
        0.0322096012532711,
        0.1948537677526474,
        0.10816236585378647,
        -0.029297366738319397,
    },
};

const gruval bias[KERNEL_COLS] = {
    -0.0007574477349407971,
    -0.013307561166584492,
    0.013961559161543846,
    0.026683257892727852,
    -0.017528248950839043,
    -0.027874281629920006,
    0.005773439072072506,
    0.024008657783269882,
    -0.05437887832522392,
    0.05094195529818535,
    0.007748863659799099,
    -0.02839399129152298,
};
const gruval recurrent_bias[KERNEL_COLS] = {
    -0.0007574477349407971,
    -0.013307561166584492,
    0.013961559161543846,
    0.026683257892727852,
    -0.017528248950839043,
    -0.027874281629920006,
    0.005773439072072506,
    0.024008657783269882,
    -0.05437887832522392,
    0.05094195529818535,
    0.007748863659799099,
    -0.02839399129152298,
};

const gruval output_expected[OUTPUT_SIZE] = {
    -0.031386073678731918,
    0.183457866311073303,
    0.134453982114791870,
    0.013991579413414001,
};



const gruval sigmoid_z_input[64] = {
    0.206601396, -0.414348811, 0.00123198703 -0.282136142,
};
const gruval sigmoid_z_output[64] = {
    0.551467419, 0.397869825, 0.500308037, 0.429930151,
};

const gruval sigmoid_r_input[64] = {
    -0.666760743, 0.0815263093, 0.503777802, 0.787091076,
};
const gruval sigmoid_r_output[64] = {
    0.339222521, 0.520370245, 0.623346686, 0.687206388,
};

const gruval tanh_hh_input[64] = {
    -0.0700895637, 0.314671963, 0.275865018, 0.0245485548,
};
const gruval tanh_hh_output[64] = {
    0.339222521, 0.520370245, 0.623346686, 0.687206388,
};
