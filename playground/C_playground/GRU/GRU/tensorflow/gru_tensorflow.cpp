#include "../utils/sigmoid.h"
#include "../utils/tanh.h"
#include "gru_tensorflow.h"

// DOT product -> 1st matrix n_cols == 2nd matrix n_rows

float matrix_x[KERNEL_COLS];
float matrix_inner[KERNEL_COLS];

float z[SPLIT_SIZE];
float r[SPLIT_SIZE];
float hh[SPLIT_SIZE];


void gru_tensorflow(float* input, float* state, float* output) {
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