#include "data.h"
#include "data_simple.h"
#include "gru_rnnoise.h"
#include "../utils/sigmoid.h"

#include <stdio.h>

#define DEBUG
#include "debug_rrnoise.h"


#define celt_isnan(x) ((x)!=(x))


//#define WEIGHTS_SCALE (1.f/256)
#define WEIGHTS_SCALE 1.f       // based on results from tensorflow sigmoid

#define MAX_NEURONS 128

#define ACTIVATION_TANH    0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_RELU    2


//typedef signed char rnn_weight;

typedef struct {
    const float* tf_bias;
    const float* input_weights;
    const float* recurrent_weights;
    int nb_inputs;
    int nb_neurons;
    int activation;
} GRULayer;




#include <math.h>  // floor
static inline float tansig_approx(float x) {
    int i;
    float y, dy;
    float sign = 1;
    /* Tests are reversed to catch NaNs */
    if (!(x < 8))
        return 1;
    if (!(x > -8))
        return -1;
#ifndef FIXED_POINT
    /* Another check in case of -ffast-math */
    if (celt_isnan(x))
        return 0;
#endif
    if (x < 0) {
        x = -x;
        sign = -1;
    }
    i = (int)floor(.5f + 25 * x);
    x -= .04f * i;
    y = tansig_table[i];
    dy = 1 - y * y;
    y = y + x * dy * (1 - y * x);
    return sign * y;
}

static inline float sigmoid_approx(float x) { return .5 + .5 * tansig_approx(.5 * x); }
//static inline float sigmoid_approx(float x) { return sigmoidf(x); }

static inline float relu(float x) { return x < 0 ? 0 : x; }


void compute_gru(const GRULayer* gru, float* state, const float* input) {
    int i, j;
    int N, M;
    int stride;
    float z[MAX_NEURONS];
    float r[MAX_NEURONS];
    float h[MAX_NEURONS];
    M = gru->nb_inputs;
    N = gru->nb_neurons;
    stride = 3 * N;
    PRINT_M_N_STRIDE(M, N, stride);
    PRINT_SEPARATOR;

    PRINT_UG_TITLE(N);
    for (i = 0; i < N; i++) {
        /* Compute update gate. */
        float sum = gru->tf_bias[i];
        PRINT_K_LOOP(M);
        for (j = 0; j < M; j++) {
            int idx = j * stride + i;
            sum += gru->input_weights[idx] * input[j];
            PRINT_K(j, sum, idx, gru->input_weights[idx], input[j]);
        }
        PRINT_RK_LOOP(N);
        for (j = 0; j < N; j++) {
            int idx = j * stride + i;
            sum += gru->recurrent_weights[idx] * state[j];
            PRINT_RK(j, sum, idx, gru->recurrent_weights[idx], state[j]);
        }
        z[i] = sigmoid_approx(WEIGHTS_SCALE * sum);
        PRINT_Z(i, sum, z[i]);
    }

    PRINT_SEPARATOR;
    PRINT_RG_TITLE(N);
    for (i = 0; i < N; i++) {
        /* Compute reset gate. */
        float sum = gru->tf_bias[N + i];
        PRINT_K_LOOP(M);
        for (j = 0; j < M; j++) {
            int idx = N + j * stride + i;
            sum += gru->input_weights[idx] * input[j];
            PRINT_K(j, sum, idx, gru->input_weights[idx], input[j]);
        }
        PRINT_RK_LOOP(N);
        for (j = 0; j < N; j++) {
            int idx = N + j * stride + i;
            sum += gru->recurrent_weights[N + j * stride + i] * state[j];
            PRINT_RK(j, sum, idx, gru->recurrent_weights[idx], state[j]);
        }
        r[i] = sigmoid_approx(WEIGHTS_SCALE * sum);
        PRINT_R(i, sum, r[i]);
    }

    PRINT_SEPARATOR;
    PRINT_CO_TITLE(N);
    for (i = 0; i < N; i++) {
        /* Compute output. */
        float sum = gru->tf_bias[2 * N + i];
        PRINT_K_LOOP(M);
        for (j = 0; j < M; j++) {
            int idx = 2 * N + j * stride + i;
            sum += gru->input_weights[idx] * input[j];
            PRINT_K(j, sum, idx, gru->input_weights[idx], input[j]);
        }
        PRINT_RK_LOOP(N);
        for (j = 0; j < N; j++) {
            int idx = 2 * N + j * stride + i;
            sum += gru->recurrent_weights[idx] * state[j] * r[j];
            PRINT_RK(j, sum, idx, gru->recurrent_weights[idx], state[j]);
        }
        /*
        if (gru->activation == ACTIVATION_SIGMOID) sum = sigmoid_approx(WEIGHTS_SCALE * sum);
        else if (gru->activation == ACTIVATION_TANH) sum = tansig_approx(WEIGHTS_SCALE * sum);
        else if (gru->activation == ACTIVATION_RELU) sum = relu(WEIGHTS_SCALE * sum);
        else *(int*)0 = 0;
        */
        sum = tansig_approx(WEIGHTS_SCALE * sum);   // ACTIVATION_TANH
        h[i] = z[i] * state[i] + (1 - z[i]) * sum;
        PRINT_H(i, sum, h[i]);
    }
    for (i = 0; i < N; i++)
        state[i] = h[i];

    printf("\n");
}





//#define INPUT_SIZE      64
//#define OUTPUT_SIZE     128

void gru_rnnoise(float* output) {
    /*
    GRULayer gruLayer = {
        *forward_bias,
        *forward_kernel,
        *forward_recurrent_kernel,
        64,
        64,
        ACTIVATION_TANH
    };
    */
    //"activation": "tanh",
    //"recurrent_activation": "sigmoid",

    //float output[OUTPUT_SIZE];
    //float input[INPUT_SIZE];

    GRULayer gruLayer = {
        *simple_bias,
        *simple_kernel,
        *simple_recurrent_kernel,
        2,
        2,
        ACTIVATION_TANH
    };

    //const float* input = gru_input;
    const float* input = simple_input;

    /*
    for (int i = 0; i < 12; ++i) {
        output[i] = simple_recurrent_initializer[i];
    }
    */

    compute_gru(&gruLayer, output, input);




    float x_z = 1.f;
    float x_r = 1.f;
    float x_h = 1.f;

    float recurrent_z = 1.f;
    float recurrent_r = 1.f;
    float recurrent_h = 1.f;

    float z = sigmoid_approx(x_z + recurrent_z);
    float r = sigmoid_approx(x_r + recurrent_r);
    float hh = tansig_approx(x_h + r * recurrent_h);
    //output[6] = hh;

    float h_tm1 = 0;
    float h = z * h_tm1 + (1 - z) * hh;
    //output[7] = h;

    //float z = sigmoid_approx(WEIGHTS_SCALE * 0);
    //float sum = sigmoid_approx(WEIGHTS_SCALE * 0);
    //float res = z * 0 + (1 - z) * sum;
    //output[7] = res;
    
    float sig = sigmoid_approx(WEIGHTS_SCALE * 2);
    output[6] = sig;
    float tanh = tansig_approx(WEIGHTS_SCALE * 2);
    output[7] = tanh;
}
