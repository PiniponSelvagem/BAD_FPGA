#include <stdio.h>
#include <string.h>

#include "input/input.h"

#include "conv2d/conv2d.h"
#include "bnorm/bnorm.h"

/* 0 */
#include "conv2d/data/conv2d_0.h"
#include "bnorm/data/bnorm_0.h"

/* 1 */
#include "conv2d/data/conv2d_1.h"
#include "bnorm/data/bnorm_1.h"




void special_print(float out, float difference, float exp) {
    char difference_str[50];
    // Use snprintf to format difference as a string with 4 digits after the decimal point
    snprintf(difference_str, sizeof(difference_str), "%.12f", difference);

    // Loop through the string and replace '0' characters with '_'
    for (int i = 0; i < strlen(difference_str); i++) {
        if (difference_str[i] == '0') {
            difference_str[i] = '_';
        }
    }

    // Print the formatted string with underscores
    printf(" %15.12f | %15s | %15.12f\n", out, difference_str, exp);
}


#define OUT_MAX_PRINT   3
#include "conv2d/data/conv2d_0_outex.h"
#include "bnorm/data/bnorm_0_outex.h"

input_t inputpad[C2D_0__IN_LINES][C2D_0__IN_COLS] = { 0 };
conv_t outpad_a[CHANNELS][C2D_0__OUT_LINES][C2D_0__OUT_COLS] = { 0 };
conv_t outpad_b[CHANNELS][C2D_0__OUT_LINES][C2D_0__OUT_COLS] = { 0 };
conv_t outpad_c[CHANNELS][C2D_0__OUT_LINES][C2D_0__OUT_COLS] = { 0 };
//conv_t output[INPUT_LINES][INPUT_COLS] = { 0 };

conv_t expt[40] = {
    0.03574496880173683,
    -0.16295084357261658,
    0.05252210050821304,
    0.06769507378339767,
    0.039424020797014236,
    0.06309892982244492,
    0.05123986303806305,
    0.039152003824710846,
    0.0446232408285141,
    0.038316741585731506,
    0.05122825503349304,
    0.048744477331638336,
    0.0562068447470665,
    0.054037392139434814,
    0.051755428314208984,
    0.05374322086572647,
    0.05435632914304733,
    0.05294995754957199,
    0.05924156308174133,
    0.046602457761764526,
    0.03196296468377113,
    0.02187073975801468,
    0.010436628013849258,
    -0.005802754312753677,
    -0.013196997344493866,
    -0.026864122599363327,
    -0.020618684589862823,
    -0.009180033579468727,
    0.015875380486249924,
    0.025483597069978714,
    0.028949663043022156,
    0.013506952673196793,
    0.01384444534778595,
    0.007258392870426178,
    0.010946892201900482,
    0.009377814829349518,
    0.010923326015472412,
    0.013069838285446167,
    -0.023294763639569283,
    -0.05424191430211067
};


void printarrayB() {
    for (int i = 0; i < 1; i++) {
        for (int j = 1; j < 5; j++) {
            for (int k = 1; k < 5; k++) {
                printf("%.4f ", outpad_b[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
void printarrayC() {
    for (int i = 0; i < 1; i++) {
        for (int j = 1; j < 5; j++) {
            for (int k = 1; k < 5; k++) {
                printf("%.4f ", outpad_c[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}






const conv_t kernel_X[2][2][C2D_KERNEL_LINES][C2D_KERNEL_COLS] = {
    {
        {
            {
                -6.630041752941906e-05,
                0.0006002355949021876,
                0.0016033230349421501
            },
            {
                0.0014374729944393039,
                -0.0017843039240688086,
                0.002143224934116006
            },
            {
                0.0018476983532309532,
                -0.0022525498643517494,
                7.375729910563678e-05
            }
        },
        {
            {
                0.005152823869138956,
                0.00027817676891572773,
                0.0005539240082725883
            },
            {
                0.004290696699172258,
                -2.8620061129913665e-05,
                -0.0005685683572664857
            },
            {
                0.005639079958200455,
                0.0007368568913079798,
                0.0006550222169607878
            }
        },
    }, // filter 0
    {
        {
            {
                -0.0006132876151241362,
                -0.0010441816411912441,
                -0.00029011836159043014
            },
            {
                0.0003125772636849433,
                0.0006222657975740731,
                0.0013449627440422773
            },
            {
                0.0007749365177005529,
                0.002413684269413352,
                0.0013106765691190958
            }
        },
        {
            {
                -0.0014728648820891976,
                -0.0018859279807657003,
                0.0003634647582657635
            },
            {
                -0.0005907678278163075,
                -0.0008779632626101375,
                0.0003079060115851462
            },
            {
                -0.0010302024893462658,
                -0.00046692180330865085,
                -4.559289664030075e-05
            }
        }
    }, // filter 1
};

const conv_t bias_X[2] = {
    -0.022464923560619354,
    -0.02606898359954357,
};

const conv_t input_X[2][7][12] = {
    {
        {
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
        }, // pad
        {   
            0,  // pad
            0,
            0.31738993525505066,
            0.2551617920398712,
            0.1682370901107788,
            0.2514021694660187,
            0.17518079280853271,
            0.16509199142456055,
            0.19018620252609253,
            0.14896106719970703,
            0.16536399722099304,
            0,  // pad
        },
        {   
            0,  // pad
            0.0,
            0.18299192190170288,
            0.05062472075223923,
            0.0,
            0.021046213805675507,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.0,
            0.21922284364700317,
            0.04227350652217865,
            0.0,
            0.07744336873292923,
            0.0,
            0.0016616657376289368,
            0.00875633955001831,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.0,
            0.25430962443351746,
            0.02633807063102722,
            0.012720633298158646,
            0.026620768010616302,
            0.020646385848522186,
            0.011813916265964508,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.0,
            0.22074884176254272,
            0.05763819068670273,
            0.0,
            0.0,
            0.005146969109773636,
            0.013648465275764465,
            0.0,
            0.0,
            0.017243601381778717,
            0,  // pad
        },
        {
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
        }, // pad
    },
    {
        {
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
        }, // pad
        {
            0,  // pad
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.505129337310791,
            0.0,
            0.0,
            0.0058834366500377655,
            0.0,
            0.02095923200249672,
            0.004131972789764404,
            0.0,
            0.006122913211584091,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.46068719029426575,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.006555091589689255,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.45915260910987854,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00492515042424202,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.4930756688117981,
            0.0,
            0.016723673790693283,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
        }, // pad
    }
};






const conv_t kernel_Y[2][2][C2D_KERNEL_LINES][C2D_KERNEL_COLS] = {
    {
        {
            {
                0,
                0,
                0
            },
            {
                0,
                1,
                0
            },
            {
                0,
                0,
                0
            }
        },
        {
            {
                0,
                0,
                0
            },
            {
                0,
                1,
                0
            },
            {
                0,
                0,
                0
            }
        }
    }, // filter 0
    {
        {
            {
                0,
                0,
                0
            },
            {
                0,
                0,
                0
            },
            {
                0,
                0,
                0
            }
        },
        {
            {
                0,
                0,
                0
            },
            {
                0,
                1,
                0
            },
            {
                0,
                0,
                0
            }
        }
    }, // filter 1
};

const conv_t bias_Y[2] = {
    1,
    2,
};

const conv_t input_Y[2][7][12] = {
    {
        {
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
        }, // pad
        {
            0,  // pad
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
        }, // pad
    },
    {
        {
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
        }, // pad
        {
            0,  // pad
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,  // pad
        },
        {
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
            0,  // pad
        }, // pad
    }
};

conv_t output_aux1[2][7][12] = { 0 };


void predict(const input_t input[INPUT_LINES][INPUT_COLS]/*, output_t output*/) {
    input_preconv2d(input, inputpad);

    /* 0 */
    for (int c = 0; c < CHANNELS; ++c) {
        conv2d<C2D_0__IN_LINES, C2D_0__IN_COLS, C2D_0__OUT_LINES, C2D_0__OUT_COLS>(inputpad, kernel_0[c], bias_0[c], outpad_a[c]);
    }
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_0__IN_LINES, BNORM_0__IN_COLS>(outpad_a[c], gamma_0[c], beta_0[c], movingmean_0[c], movingvariance_0[c], outpad_b[c]);
    }

    /* 1 */
    for (int c = 0; c < CHANNELS; ++c) {
        conv_t biasVal = bias_1[c];
        int f = 0;
        conv2d<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_b[f], kernel_1[f][c], biasVal, outpad_c[c]);
        ++f;
        for (; f < FILTERS; ++f) {
            conv2d_multi<C2D_1__IN_LINES, C2D_1__IN_COLS, C2D_1__OUT_LINES, C2D_1__OUT_COLS>(outpad_b[f], outpad_c[c], kernel_1[f][c], 0, outpad_c[c]);
        }
    }
    for (int c = 0; c < CHANNELS; ++c) {    // BATCH NORMALIZATION + RELU
        bnorm<BNORM_1__IN_LINES, BNORM_1__IN_COLS>(outpad_c[c], gamma_1[c], beta_1[c], movingmean_1[c], movingvariance_1[c], outpad_a[c]);
    }




    printf("");

    /*
    // Print the output values
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int i = 1; i < (40+1); ++i) {
        bnorm_t out = outpad_c[0][1][i];
        bnorm_t exp = expt[i - 1];
        bnorm_t difference = out - exp;
        //printf(" %15.12f | %7.4f | %15.12f\n", out, difference, exp);
        special_print(out, difference, exp);
    }
    */

    /*
    // Print the output values
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int outter = 1; outter < OUT_MAX_PRINT; ++outter) {
        printf("[%3d] ##############################################\n", outter);
        for (int i = 1; i < INPUT_COLS; ++i) {
            bnorm_t out = outputpad[0][outter][i];
            bnorm_t exp = bnorm_output_expected_channel0[outter - 1][i - 1];
            bnorm_t difference = out - exp;
            //printf(" %15.12f | %7.4f | %15.12f\n", out, difference, exp);
            special_print(out, difference, exp);
        }
        printf("\n");
    }
    */
}

conv_t test_in[2][6][6] = {
        {
            {0, 0,0,0,0, 0},

            {0, 0,0,0,0, 0},
            {0, 0,2,0,0, 0},
            {0, 0,0,0,0, 0},
            {0, 0,0,0,0, 0},

            {0, 0,0,0,0, 0},
        },
        {
            {0, 0,0,0,0, 0},

            {0, 0,0,0,0, 0},
            {0, 0,1,0,0, 0},
            {0, 0,0,0,0, 0},
            {0, 0,0,0,0, 0},

            {0, 0,0,0,0, 0},
        },
};
conv_t test_bias[2] = { 1, 0 };
conv_t test_weights[2][2][3][3] = {
    {
        {
            {0,0,0},
            {0,2,0},
            {0,0,0},
        },
        {
            {0,0,0},
            {0,4,0},
            {0,0,0},
        }
    },
    {
        {
            {0,0,0},
            {0,0,0},
            {0,0,0},
        },
        {
            {0,0,0},
            {0,0,0},
            {0,0,0},
        }
    }
};

conv_t test_out_aux[2][6][6] = { 0 };
conv_t test_out[2][6][6] = { 0 };

void printarray() {
    for (int i = 0; i < 1; i++) {
        for (int j = 1; j < 5; j++) {
            for (int k = 1; k < 5; k++) {
                printf("%.4f ", test_out_aux[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


int main() {

    predict(input);

    /*
    for (int c = 0; c < 2; ++c) {
        conv_t biasVal = bias_Y[c];
        int f = 0;
        conv2d<7, 12, 7, 12>(input_Y[f], kernel_Y[f][c], biasVal, output_aux1[c]);
        ++f;
        for (; f < 2; ++f) {
            conv2d_multi<7, 12, 7, 12>(input_Y[f], output_aux1[c], kernel_Y[f][c], 0, output_aux1[c]);
        }
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 1; j < 5; j++) {
            for (int k = 1; k < 10; k++) {
                printf("%10.6f ", output_aux1[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */




    /*
    for (int f = 0; f < 2; ++f) {
        conv_t biasVal = test_bias[f];
        conv2d<6, 6, 6, 6>(test_in[0], test_weights[f][0], biasVal, test_out_aux[f]);
        printarray();
        for (int c = 1; c < 2; ++c) {
            conv2d_multi<6, 6, 6, 6>(test_in[c], test_out_aux[f], test_weights[f][c], 0, test_out_aux[f]);
            printarray();
        }
        break;
    }
    */



    /*
    input_preconv2d(input, inputpad);

    conv2d<C2D_0__IN_LINES, C2D_0__IN_COLS, C2D_0__OUT_LINES, C2D_0__OUT_COLS>(inputpad, kernel_0[0], bias_0[0], outputpad);
    conv2d_postprocess(outputpad, output);

    // Print the output values
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int outter = 1; outter < OUT_MAX_PRINT; ++outter) {
        printf("[%3d] ##############################################\n", outter);
        for (int i = 1; i < (INPUT_COLS+1); ++i) {
            conv_t out = outputpad[outter][i];
            conv_t exp = c2d_output_expected_channel0[outter-1][i-1];
            conv_t difference = out - exp;
            //printf(" %15.12f | %7.4f | %15.12f\n", out, difference, exp);
            special_print(out, difference, exp);
        }
        printf("\n");
    }

    printf("#############################################################################\n");
    bnorm<BNORM_0__IN_LINES, BNORM_0__IN_COLS>(outputpad, gamma_0[0], beta_0[0], movingmean_0[0], movingvariance_0[0], outputpad);

    // Print the output values
    printf("      OUTPUT            --VS--            EXPECTED\n");
    for (int outter = 1; outter < OUT_MAX_PRINT; ++outter) {
        printf("[%3d] ##############################################\n", outter);
        for (int i = 1; i < INPUT_COLS; ++i) {
            bnorm_t out = outputpad[outter][i];
            bnorm_t exp = bnorm_output_expected_channel0[outter-1][i-1];
            bnorm_t difference = out - exp;
            //printf(" %15.12f | %7.4f | %15.12f\n", out, difference, exp);
            special_print(out, difference, exp);
        }
        printf("\n");
    }
    */

    return 0;
}
