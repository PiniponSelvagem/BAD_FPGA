#pragma once

const float simple_kernel[2][6] = {
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
};

const float simple_recurrent_kernel[2][6] = {
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
};

const float simple_bias[2][6] = {
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
};


const float simple_recurrent_initializer[12] = {
      0.2126342, 0.81283575, 0.49438077,  0.21087615,  0.04209568, -0.05857829,
     -0.4336365, 0.11620823, 0.34572333, -0.68314976, -0.41594738,  0.19805196
};

const float simple_input[2] = {
    1, 1
};

const float simple_output_expected[2] = {
    0.01797373, 0.01797373
};