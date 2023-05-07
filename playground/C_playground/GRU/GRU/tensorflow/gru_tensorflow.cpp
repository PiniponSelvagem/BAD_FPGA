#include "gru_tensorflow.h"

#include <stdio.h>
#include <string.h>

//#define DEBUG_PRINT
#include "debug_print.h"
#define FORMAT "%.9f "


// DOT product -> 1st matrix n_cols == 2nd matrix n_rows

gruval matrix_x[KERNEL_COLS];
gruval matrix_inner[KERNEL_COLS];

gruval z[SPLIT_SIZE];
gruval r[SPLIT_SIZE];
gruval hh[SPLIT_SIZE];

#define DEBUG_S_T
#ifdef DEBUG_S_T
    #define DEBUG_SAVE(arr, value, i) arr[i] = value;
#else
    #define DEBUG_SAVE(arr, value, i) ;
#endif
gruval z_in[SPLIT_SIZE];
gruval r_in[SPLIT_SIZE];
gruval hh_in[SPLIT_SIZE];

void compare_print(gruval out, gruval difference, gruval exp) {
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

void compare(const char* title, gruval* out, const gruval* exp) {
    printf(title);
    for (int i = 0; i < SPLIT_SIZE; i++) {
        gruval out_val = out[i];
        gruval exp_val = exp[i];
        gruval difference = out_val - exp_val;
        compare_print(out_val, difference, exp_val);
    }
    printf("\n");
}

void test_sig_z_values() {
    compare("      Z INPUT           --VS--           EXPECTED\n", z_in, sigmoid_z_input);
    compare("      Z OUTPUT          --VS--           EXPECTED\n", z, sigmoid_z_output);
}
void test_sig_r_values() {
    compare("      R INPUT           --VS--           EXPECTED\n", r_in, sigmoid_r_input);
    compare("      R OUTPUT          --VS--           EXPECTED\n", r, sigmoid_r_output);
}
void test_tanh_hh_values() {
    compare("      HH INPUT          --VS--           EXPECTED\n", hh_in, tanh_hh_input);
    compare("      HH OUTPUT         --VS--           EXPECTED\n", hh, tanh_hh_output);
}


void gru_tensorflow(gruval* input, gruval* state, gruval* output) {
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
            gruval iVal = input[j];
            gruval kVal = kernel[j][i];
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
    gruval* x_z = matrix_x + (SPLIT_SIZE * 0);
    gruval* x_r = matrix_x + (SPLIT_SIZE * 1);
    gruval* x_h = matrix_x + (SPLIT_SIZE * 2);
    PRINT_ARRAY_1D("x_z", x_z, SPLIT_SIZE);
    PRINT_ARRAY_1D("x_r", x_r, SPLIT_SIZE);
    PRINT_ARRAY_1D("x_h", x_h, SPLIT_SIZE);

    /* hidden state projected by all gate matrices at once */
    // matrix_inner = backend.dot(h_tm1, recurrent_kernel)
    for (int i = 0; i < KERNEL_COLS; ++i) {
        matrix_inner[i] = 0;
        for (int j = 0; j < KERNEL_ROWS; ++j) {
            gruval iVal = state[j];
            gruval kVal = recurrent_kernel[j][i];
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
    gruval* recurrent_z = matrix_inner + (SPLIT_SIZE * 0);
    gruval* recurrent_r = matrix_inner + (SPLIT_SIZE * 1);
    gruval* recurrent_h = matrix_inner + (SPLIT_SIZE * 2);
    PRINT_ARRAY_1D("recurrent_z", recurrent_z, SPLIT_SIZE);
    PRINT_ARRAY_1D("recurrent_r", recurrent_r, SPLIT_SIZE);
    PRINT_ARRAY_1D("recurrent_h", recurrent_h, SPLIT_SIZE);

    // z = tf.sigmoid(x_z + recurrent_z)
    for (int i = 0; i < SPLIT_SIZE; ++i) {
        gruval value = x_z[i] + recurrent_z[i];
        DEBUG_SAVE(z_in, value, i)
        z[i] = SIGMOID(value);
    }
    PRINT_ARRAY_1D_X("x_z + recurrent_z", z_in, SPLIT_SIZE, FORMAT);
    PRINT_ARRAY_1D_X("z", z, SPLIT_SIZE, FORMAT);
    // r = tf.sigmoid(x_r + recurrent_r)
    for (int i = 0; i < SPLIT_SIZE; ++i) {
        gruval value = x_r[i] + recurrent_r[i];
        DEBUG_SAVE(r_in, value, i)
        r[i] = SIGMOID(value);
    }
    PRINT_ARRAY_1D_X("x_r + recurrent_r", r_in, SPLIT_SIZE, FORMAT);
    PRINT_ARRAY_1D_X("r", r, SPLIT_SIZE, FORMAT);
    // hh = tf.tanh(x_h + r * recurrent_h)
    for (int i = 0; i < SPLIT_SIZE; ++i) {
        gruval value = x_h[i] + (r[i] * recurrent_h[i]);
        DEBUG_SAVE(hh_in, value, i)
        hh[i] = TANH(value);
    }
    PRINT_ARRAY_1D_X("x_h + recurrent_h", hh_in, SPLIT_SIZE, FORMAT);
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
void gru_tensorflow_clean(gruval* input, gruval* state, gruval* output) {
    // state = (h_tm1 = cell_states[0])

    /* inputs projected by all gate matrices at once */
    // matrix_x = backend.dot(cell_inputs, kernel)
    for (int i = 0; i < KERNEL_COLS; ++i) {
        matrix_x[i] = 0;
        for (int j = 0; j < KERNEL_ROWS; ++j) {
            gruval iVal = input[j];
            gruval kVal = kernel[j][i];
            matrix_x[i] += iVal * kVal;
        }
    }
    // matrix_x = backend.bias_add(matrix_x, input_bias)
    for (int i = 0; i < KERNEL_COLS; ++i) {
        matrix_x[i] += bias[i];
    }

    // x_z, x_r, x_h = tf.split(matrix_x, 3, axis=1)
    gruval* x_z = matrix_x + (SPLIT_SIZE * 0);
    gruval* x_r = matrix_x + (SPLIT_SIZE * 1);
    gruval* x_h = matrix_x + (SPLIT_SIZE * 2);

    /* hidden state projected by all gate matrices at once */
    // matrix_inner = backend.dot(h_tm1, recurrent_kernel)
    for (int i = 0; i < KERNEL_COLS; ++i) {
        matrix_inner[i] = 0;
        for (int j = 0; j < KERNEL_ROWS; ++j) {
            gruval iVal = state[j];
            gruval kVal = recurrent_kernel[j][i];
            matrix_inner[i] += iVal * kVal;
        }
    }
    // matrix_x = backend.bias_add(matrix_inner, recurrent_bias)
    for (int i = 0; i < KERNEL_COLS; ++i) {
        matrix_inner[i] += recurrent_bias[i];
    }
    // recurrent_z, recurrent_r, recurrent_h = tf.split(matrix_inner, 3, axis=1
    gruval* recurrent_z = matrix_inner + (SPLIT_SIZE * 0);
    gruval* recurrent_r = matrix_inner + (SPLIT_SIZE * 1);
    gruval* recurrent_h = matrix_inner + (SPLIT_SIZE * 2);

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
        hh[i] = tanh(x_h[i] + (r[i] * recurrent_h[i]));
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