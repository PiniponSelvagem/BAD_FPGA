#include "../utils/sigmoid.h"
#include "../utils/tanh.h"
#include "gru_tensorflow.h"

#define DEBUG
#include "debug_print.h"
#define FORMAT "%.9f "


// DOT product -> 1st matrix n_cols == 2nd matrix n_rows

float matrix_x[KERNEL_COLS];
float matrix_inner[KERNEL_COLS];

float z[SPLIT_SIZE];
float r[SPLIT_SIZE];
float hh[SPLIT_SIZE];


void gru_tensorflow(float* input, float* state, float* output) {
    PRINT_STRING("################ STEP START ################");
    PRINT_ARRAY_2D("kernel", kernel, KERNEL_ROWS, KERNEL_COLS);
    PRINT_ARRAY_2D("recurrent_kernel", recurrent_kernel, KERNEL_ROWS, KERNEL_COLS);
    PRINT_ARRAY_1D("bias", bias, BIAS_COLS);
    PRINT_ARRAY_1D("recurrent_bias", recurrent_bias, BIAS_COLS);
    PRINT_ARRAY_1D("input", input, INPUT_SIZE);
    PRINT_ARRAY_1D("state (h_tm1)", state, INPUT_SIZE);     // STATE_SIZE == INPUT_SIZE ???????????????????????
    PRINT_STRING("--------------------------------------------");
    // state = (h_tm1 = cell_states[0])

    /* inputs projected by all gate matrices at once */
    // matrix_x = backend.dot(cell_inputs, kernel)
    for (int i = 0; i < KERNEL_COLS; ++i) {
        matrix_x[i] = 0;
        for (int j = 0; j < KERNEL_ROWS; ++j) {
            float iVal = input[j];
            float kVal = kernel[j][i];
            matrix_x[i] += iVal * kVal;
        }
    }
    PRINT_ARRAY_1D("matrix_x (dot)", matrix_x, KERNEL_COLS);
    // matrix_x = backend.bias_add(matrix_x, input_bias)
    for (int i = 0; i < KERNEL_COLS; ++i) {     
        matrix_x[i] += bias[i];
    }
    PRINT_ARRAY_1D("matrix_x (bias_add)", matrix_x, KERNEL_COLS);

    // x_z, x_r, x_h = tf.split(matrix_x, 3, axis=1)
    float* x_z = matrix_x + (SPLIT_SIZE * 0);
    float* x_r = matrix_x + (SPLIT_SIZE * 1);
    float* x_h = matrix_x + (SPLIT_SIZE * 2);
    PRINT_ARRAY_1D("x_z", x_z, SPLIT_SIZE);
    PRINT_ARRAY_1D("x_r", x_r, SPLIT_SIZE);
    PRINT_ARRAY_1D("x_h", x_h, SPLIT_SIZE);

    /* hidden state projected by all gate matrices at once */
    // matrix_inner = backend.dot(h_tm1, recurrent_kernel)
    for (int i = 0; i < KERNEL_COLS; ++i) {
        matrix_inner[i] = 0;
        for (int j = 0; j < KERNEL_ROWS; ++j) {
            float iVal = state[j];
            float kVal = recurrent_kernel[j][i];
            matrix_inner[i] += iVal * kVal;
        }
    }
    PRINT_ARRAY_1D("matrix_inner (dot)", matrix_inner, KERNEL_COLS);
    // matrix_x = backend.bias_add(matrix_inner, recurrent_bias)
    for (int i = 0; i < KERNEL_COLS; ++i) {
        matrix_inner[i] += recurrent_bias[i];
    }
    PRINT_ARRAY_1D("matrix_inner (bias_add)", matrix_inner, KERNEL_COLS);
    // recurrent_z, recurrent_r, recurrent_h = tf.split(matrix_inner, 3, axis=1
    float* recurrent_z = matrix_inner + (SPLIT_SIZE * 0);
    float* recurrent_r = matrix_inner + (SPLIT_SIZE * 1);
    float* recurrent_h = matrix_inner + (SPLIT_SIZE * 2);
    PRINT_ARRAY_1D("recurrent_z", recurrent_z, SPLIT_SIZE);
    PRINT_ARRAY_1D("recurrent_r", recurrent_r, SPLIT_SIZE);
    PRINT_ARRAY_1D("recurrent_h", recurrent_h, SPLIT_SIZE);

    // z = tf.sigmoid(x_z + recurrent_z)
    for (int i = 0; i < SPLIT_SIZE; ++i) {
        z[i] = sigmoid(x_z[i] + recurrent_z[i]);
    }
    PRINT_ARRAY_1D_X("z", z, SPLIT_SIZE, FORMAT);
    // r = tf.sigmoid(x_r + recurrent_r)
    for (int i = 0; i < SPLIT_SIZE; ++i) {
        r[i] = sigmoid(x_r[i] + recurrent_r[i]);
    }
    PRINT_ARRAY_1D_X("r", r, SPLIT_SIZE, FORMAT);
    // hh = tf.tanh(x_h + r * recurrent_h)
    for (int i = 0; i < SPLIT_SIZE; ++i) {
        hh[i] = tanh(x_h[i] + (r[i] * recurrent_z[i]));
    }
    PRINT_ARRAY_1D_X("hh", hh, SPLIT_SIZE, FORMAT);

    // previous and candidate state mixed by update gate
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] = z[i] * state[i] + (1 - z[i]) * hh[i];
    }
    PRINT_ARRAY_1D_X("output (h)", output, OUTPUT_SIZE, FORMAT);

    PRINT_STRING("################ STEP END #################");
    PRINT_SEPARATOR;
}




/**
 * ONLY FOR BETTER VISUALIZATION!!!
 * CHANGES SHOULD ONLY BE MADE IN: gru_tensorflow
 */
#ifdef NO_PRINTS
void gru_tensorflow_clean(float* input, float* state, float* output) {
    // state = (h_tm1 = cell_states[0])

    /* inputs projected by all gate matrices at once */
    // matrix_x = backend.dot(cell_inputs, kernel)
    for (int i = 0; i < KERNEL_COLS; ++i) {
        matrix_x[i] = 0;
        for (int j = 0; j < KERNEL_ROWS; ++j) {
            float iVal = input[j];
            float kVal = kernel[j][i];
            matrix_x[i] += iVal * kVal;
        }
    }
    // matrix_x = backend.bias_add(matrix_x, input_bias)
    for (int i = 0; i < KERNEL_COLS; ++i) {
        matrix_x[i] += bias[i];
    }

    // x_z, x_r, x_h = tf.split(matrix_x, 3, axis=1)
    float* x_z = matrix_x + (SPLIT_SIZE * 0);
    float* x_r = matrix_x + (SPLIT_SIZE * 1);
    float* x_h = matrix_x + (SPLIT_SIZE * 2);

    /* hidden state projected by all gate matrices at once */
    // matrix_inner = backend.dot(h_tm1, recurrent_kernel)
    for (int i = 0; i < KERNEL_COLS; ++i) {
        matrix_inner[i] = 0;
        for (int j = 0; j < KERNEL_ROWS; ++j) {
            float iVal = state[j];
            float kVal = recurrent_kernel[j][i];
            matrix_inner[i] += iVal * kVal;
        }
    }
    // matrix_x = backend.bias_add(matrix_inner, recurrent_bias)
    for (int i = 0; i < KERNEL_COLS; ++i) {
        matrix_inner[i] += recurrent_bias[i];
    }
    // recurrent_z, recurrent_r, recurrent_h = tf.split(matrix_inner, 3, axis=1
    float* recurrent_z = matrix_inner + (SPLIT_SIZE * 0);
    float* recurrent_r = matrix_inner + (SPLIT_SIZE * 1);
    float* recurrent_h = matrix_inner + (SPLIT_SIZE * 2);

    // z = tf.sigmoid(x_z + recurrent_z)
    for (int i = 0; i < SPLIT_SIZE; ++i) {
        z[i] = sigmoid(x_z[i] + recurrent_z[i]);
    }
    // r = tf.sigmoid(x_r + recurrent_r)
    for (int i = 0; i < SPLIT_SIZE; ++i) {
        r[i] = sigmoid(x_r[i] + recurrent_r[i]);
    }
    // hh = tf.tanh(x_h + r * recurrent_h)
    for (int i = 0; i < SPLIT_SIZE; ++i) {
        hh[i] = tanh(x_h[i] + (r[i] * recurrent_z[i]));
    }

    // previous and candidate state mixed by update gate
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] = z[i] * state[i] + (1 - z[i]) * hh[i];
    }
}
#endif

#ifdef PYTHON
    // state == h_tm1

    //# inputs projected by all gate matrices at once
    float* matrix_x = backend.dot(cell_inputs, kernel);
    matrix_x = backend.bias_add(matrix_x, input_bias);

    x_z, x_r, x_h = tf.split(matrix_x, 3, axis = 1);

    // hidden state projected by all gate matrices at once
    float* matrix_inner = backend.dot(state, recurrent_kernel);
    matrix_inner = backend.bias_add(matrix_inner, recurrent_bias);

    recurrent_z, recurrent_r, recurrent_h = tf.split(
        matrix_inner, 3, axis = 1
    );
    float* z = tf.sigmoid(x_z + recurrent_z);
    float* r = tf.sigmoid(x_r + recurrent_r);
    float* hh = tf.tanh(x_h + r * recurrent_h);

    // previous and candidate state mixed by update gate
    float* h = z * state + (1 - z) * hh;
#endif