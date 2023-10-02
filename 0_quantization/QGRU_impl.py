if self.implementation == 1:
    inputs_z = inputs
    inputs_r = inputs
    inputs_h = inputs

    x_z = K.dot(inputs_z, quantized_kernel[:, :self.units])
    x_r = K.dot(inputs_r, quantized_kernel[:, self.units:self.units * 2])
    x_h = K.dot(inputs_h, quantized_kernel[:, self.units * 2:])

    x_z = K.bias_add(x_z, input_bias[:self.units])
    x_r = K.bias_add(x_r, input_bias[self.units: self.units * 2])
    x_h = K.bias_add(x_h, input_bias[self.units * 2:])
    
    h_tm1_z = h_tm1
    h_tm1_r = h_tm1
    h_tm1_h = h_tm1

    recurrent_z = K.dot(h_tm1_z, quantized_recurrent[:, :self.units])
    recurrent_r = K.dot(h_tm1_r, quantized_recurrent[:, self.units:self.units * 2])
    
    z = self.recurrent_activation(x_z + recurrent_z)
    r = self.recurrent_activation(x_r + recurrent_r)

    recurrent_h = K.dot(r * h_tm1_h, quantized_recurrent[:, self.units * 2:])

    hh = self.activation(x_h + recurrent_h)
else:
    # inputs projected by all gate matrices at once
    matrix_x = K.dot(inputs, quantized_kernel)
    print_tensor(matrix_x, "matrix_x (dot) =")

    # biases: bias_z_i, bias_r_i, bias_h_i
    matrix_x = K.bias_add(matrix_x, input_bias)
    print_tensor(matrix_x, "matrix_x (bias_add) =")

    x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=-1)

    matrix_inner = K.dot(h_tm1, quantized_recurrent)
    matrix_inner = K.bias_add(matrix_inner, recurrent_bias)

    recurrent_z, recurrent_r, recurrent_h = array_ops.split(matrix_inner, [self.units, self.units, -1], axis=-1)
  
    z = self.recurrent_activation(x_z + recurrent_z)
    r = self.recurrent_activation(x_r + recurrent_r)
    recurrent_h = r * recurrent_h
  
    hh = self.activation(x_h + recurrent_h)
    
# previous and candidate state mixed by update gate
h = z * h_tm1 + (1 - z) * hh
return h, [h]