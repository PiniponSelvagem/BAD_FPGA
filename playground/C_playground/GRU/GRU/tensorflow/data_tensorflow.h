#pragma once
#define INPUT_SIZE  2
#define KERNEL_ROWS INPUT_SIZE
#define KERNEL_COLS (INPUT_SIZE*3)
#define SPLIT_SIZE (KERNEL_COLS/3)
#define OUTPUT_SIZE INPUT_SIZE

#ifdef OPTION_DEBUG
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
const float kernel[KERNEL_ROWS][KERNEL_COLS] = {
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
};
const float recurrent_kernel[KERNEL_ROWS][KERNEL_COLS] = {
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
};

const float bias[KERNEL_COLS] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
const float recurrent_bias[KERNEL_COLS] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };


const float output_expected[OUTPUT_SIZE] = { 0.11378668, 0.11378668 };

#else
const float kernel[2][6] = {
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
};

const float recurrent_kernel[2][6] = {
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
};

const float bias[1][6] = {
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
};

const float recurrent_bias[1][6] = {
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
};


const float output_expected[2] = { 0.01797373, 0.01797373 }; // NOT VALID ATM

#endif
