#pragma once


#ifdef OPTION_DEBUG
    #define INPUT_SIZE  2
    #define KERNEL_ROWS INPUT_SIZE
    #define KERNEL_COLS (INPUT_SIZE*3)
    #define BIAS_COLS   KERNEL_COLS
    #define SPLIT_SIZE  (KERNEL_COLS/3)
    #define OUTPUT_SIZE INPUT_SIZE

    const float kernel[KERNEL_ROWS][KERNEL_COLS] = {
        { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 },
        { 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 }
    };
    const float recurrent_kernel[KERNEL_ROWS][KERNEL_COLS] = {
        { 2.0, 2.1, 2.2, 2.3, 2.4, 2.5 },
        { 3.0, 3.1, 3.2, 3.3, 3.4, 3.5 }
    };

    const float bias[KERNEL_COLS] = { 4.0, 4.1, 4.2, 4.3, 4.4, 4.5 };
    const float recurrent_bias[KERNEL_COLS] = { 5.0, 5.1, 5.2, 5.3, 5.4, 5.5 };


    const float output_expected[OUTPUT_SIZE] = { 1, 1 }; // NOT VALID ATM

#elif OPTION_1
    #define INPUT_SIZE  2
    #define KERNEL_ROWS INPUT_SIZE
    #define KERNEL_COLS (INPUT_SIZE*3)
    #define BIAS_COLS   KERNEL_COLS
    #define SPLIT_SIZE  (KERNEL_COLS/3)
    #define OUTPUT_SIZE INPUT_SIZE

    const float kernel[KERNEL_ROWS][KERNEL_COLS] = {
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
    };
    const float recurrent_kernel[KERNEL_ROWS][KERNEL_COLS] = {
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
    };

    const float bias[BIAS_COLS] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    const float recurrent_bias[BIAS_COLS] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };


    const float output_expected[OUTPUT_SIZE] = { 0.11378668, 0.11378668 };

#elif OPTION_2
    #define INPUT_SIZE  2
    #define KERNEL_ROWS INPUT_SIZE
    #define KERNEL_COLS (INPUT_SIZE*3)
    #define BIAS_COLS   KERNEL_COLS
    #define SPLIT_SIZE  (KERNEL_COLS/3)
    #define OUTPUT_SIZE INPUT_SIZE

    const float kernel[KERNEL_ROWS][KERNEL_COLS] = {
        {  1,  2,  3,  4,  5,  6 },
        {  7,  8,  9, 10, 11, 12 }
    };
    const float recurrent_kernel[KERNEL_ROWS][KERNEL_COLS] = {
        {  -1,  -2,  -3,  -4,  -5,  -6 },
        {  -7,  -8,  -9, -10, -11, -12 }
    };

    const float bias[KERNEL_COLS] = { 1, 1, 1, 1, 1, 1 };
    const float recurrent_bias[KERNEL_COLS] = { 2, 2, 2, 2, 2, 2 };


    const float output_expected[OUTPUT_SIZE] = { 0.000335108576, 0.000906429545 };

#elif OPTION_3
#define INPUT_SIZE  4
#define KERNEL_ROWS INPUT_SIZE
#define KERNEL_COLS (INPUT_SIZE*3)
#define BIAS_COLS   KERNEL_COLS
#define SPLIT_SIZE  (KERNEL_COLS/3)
#define OUTPUT_SIZE INPUT_SIZE

const float kernel[KERNEL_ROWS][KERNEL_COLS] = {
    { 1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2 },
    { 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1 },
    { 1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2 },
    { 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3 },
};
const float recurrent_kernel[KERNEL_ROWS][KERNEL_COLS] = {
    { 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1 },
    { 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0 },
    { 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1 },
    { 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2 },
};

const float bias[KERNEL_COLS] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
const float recurrent_bias[KERNEL_COLS] = { 0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11 };


const float output_expected[OUTPUT_SIZE] = { 0.23866725, 0.03992534, 0.00506663, 0.00064862 };


#else

/*
To set OPTION_X of data_tensorflow.h:
1. Solution Explorer, right-click on your project and select "Properties" from the context menu.
2. "Configuration Properties" > "C/C++" > "Preprocessor" from the left-hand side menu.
3. "Preprocessor Definitions" field, add your macro definition in the following format: OPTION_X
*/

#endif
