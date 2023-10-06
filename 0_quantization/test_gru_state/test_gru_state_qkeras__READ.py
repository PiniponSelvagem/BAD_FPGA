

"""
OUTPUT OF "test_gru_state_qkeras.py" SHOULD BE REDIRECTED TO "TEST_GRU_STATE.log"
THEN DELETE ANY LINE BEFORE THE 1ST:
"1/1 [==============================] - ETA: 0s"

ALSO, THIS ONLY WORKS IF THIS CODE BELOW IS REPLACED IN qrecurrent.py OF QKERAS:

#######################################################
  def call(self, inputs, states, training=None):
    if PINI_DEBUG:
        print("activation: "+str(self.activation))
        print("recurrent_activation: "+str(self.recurrent_activation))
    def print_tensor(x, message="", summarize=3):
        if PINI_DEBUG == 1:
            import sys
            def get_graph():
                if tf.executing_eagerly():
                    global _GRAPH
                    if not getattr(_GRAPH, "graph", None):
                        _GRAPH.graph = tf.__internal__.FuncGraph("keras_graph")
                    return _GRAPH.graph
                else:
                    return tf.compat.v1.get_default_graph()
            if isinstance(x, tf.Tensor) and hasattr(x, "graph"):
                print(x)
                with get_graph().as_default():
                    op = tf.print(
                        message, x, output_stream=sys.stdout, summarize=summarize
                    )
                    with tf.control_dependencies([op]):
                        return tf.identity(x)
            else:
                tf.print(message, x, output_stream=sys.stdout, summarize=summarize)
                return x
    
    # previous memory
    h_tm1_tmp = states[0] if nest.is_sequence(states) else states

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1_tmp, training, count=3)

    if self.state_quantizer:
      h_tm1 = self.state_quantizer_internal(h_tm1_tmp)
      print("STATE_QUANTIZER:", self.state_quantizer)
      print("STATE_QUANTIZER_INTERNAL:", self.state_quantizer_internal)
    else:
      h_tm1 = h_tm1_tmp

    if self.kernel_quantizer:
      quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel
    if self.recurrent_quantizer:
      quantized_recurrent = self.recurrent_quantizer_internal(self.recurrent_kernel)
    else:
      quantized_recurrent = self.kernel

    if self.use_bias:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias

      if not self.reset_after:
        input_bias, recurrent_bias = quantized_bias, None
      else:
        input_bias, recurrent_bias = array_ops.unstack(quantized_bias)

    if self.implementation == 1:
      if 0. < self.dropout < 1.:
        inputs_z = inputs * dp_mask[0]
        inputs_r = inputs * dp_mask[1]
        inputs_h = inputs * dp_mask[2]
      else:
        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

      x_z = K.dot(inputs_z, quantized_kernel[:, :self.units])
      x_r = K.dot(inputs_r, quantized_kernel[:, self.units:self.units * 2])
      x_h = K.dot(inputs_h, quantized_kernel[:, self.units * 2:])

      if self.use_bias:
        x_z = K.bias_add(x_z, input_bias[:self.units])
        x_r = K.bias_add(x_r, input_bias[self.units: self.units * 2])
        x_h = K.bias_add(x_h, input_bias[self.units * 2:])

      if 0. < self.recurrent_dropout < 1.:
        h_tm1_z = h_tm1 * rec_dp_mask[0]
        h_tm1_r = h_tm1 * rec_dp_mask[1]
        h_tm1_h = h_tm1 * rec_dp_mask[2]
      else:
        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

      recurrent_z = K.dot(h_tm1_z, quantized_recurrent[:, :self.units])
      recurrent_r = K.dot(h_tm1_r,
                          quantized_recurrent[:, self.units:self.units * 2])
      if self.reset_after and self.use_bias:
        recurrent_z = K.bias_add(recurrent_z, recurrent_bias[:self.units])
        recurrent_r = K.bias_add(recurrent_r,
                                 recurrent_bias[self.units:self.units * 2])

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      # reset gate applied after/before matrix multiplication
      if self.reset_after:
        recurrent_h = K.dot(h_tm1_h, quantized_recurrent[:, self.units * 2:])
        if self.use_bias:
          recurrent_h = K.bias_add(recurrent_h, recurrent_bias[self.units * 2:])
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = K.dot(r * h_tm1_h,
                            quantized_recurrent[:, self.units * 2:])

      hh = self.activation(x_h + recurrent_h)
    else:
      #print_tensor(None, "################ STEP START ################")
      #print_tensor(quantized_kernel, "quantized_kernel =")
      #print_tensor(quantized_recurrent, "quantized_recurrent (kernel) =")
      #print_tensor(input_bias, "input_bias =")
      #print_tensor(recurrent_bias, "recurrent_bias =")
      #print_tensor(inputs, "inputs =")
      print_tensor(h_tm1, "h_tm1 =", -1)
      #print_tensor(None, "-------------------------------------------")
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]

      # inputs projected by all gate matrices at once
      matrix_x = K.dot(inputs, quantized_kernel)
      #print_tensor(matrix_x, "matrix_x (dot) =")
      if self.use_bias:
        # biases: bias_z_i, bias_r_i, bias_h_i
        matrix_x = K.bias_add(matrix_x, input_bias)
        #print_tensor(matrix_x, "matrix_x (bias_add) =")

      x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=-1)
      #print_tensor(x_z, "x_z =")
      #print_tensor(x_r, "x_r =")
      #print_tensor(x_h, "x_h =")

      if self.reset_after:
        # hidden state projected by all gate matrices at once
        matrix_inner = K.dot(h_tm1, quantized_recurrent)
        #print_tensor(matrix_inner, "matrix_inner (dot) =")
        if self.use_bias:
          matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
          #print_tensor(matrix_inner, "matrix_inner (bias_add) =")
      else:
        # hidden state projected separately for update/reset and new
        matrix_inner = K.dot(h_tm1, quantized_recurrent[:, :2 * self.units])

      recurrent_z, recurrent_r, recurrent_h = array_ops.split(
          matrix_inner, [self.units, self.units, -1], axis=-1)
      #print_tensor(recurrent_z, "recurrent_z =")
      #print_tensor(recurrent_r, "recurrent_r =")
      #print_tensor(recurrent_h, "recurrent_h =")
      
      #print_tensor(x_z + recurrent_z, "x_z + recurrent_z =")
      z = self.recurrent_activation(x_z + recurrent_z)
      #print_tensor(z, "z =")
      #print_tensor(x_r + recurrent_r, "x_r + recurrent_r =")
      r = self.recurrent_activation(x_r + recurrent_r)
      #print_tensor(r, "r =")

      if self.reset_after:
        #print_tensor(r * recurrent_h, "r * recurrent_h =")
        recurrent_h = r * recurrent_h
        #print_tensor(recurrent_h, "recurrent_h =")
      else:
        recurrent_h = K.dot(r * h_tm1, quantized_recurrent[:, 2 * self.units:])
      
      #print_tensor(x_h + recurrent_h, "x_h + recurrent_h =")
      hh = self.activation(x_h + recurrent_h)
      #print_tensor(hh, "hh =")
    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    print_tensor(h, "h =", -1)
    #print_tensor(None, "################ STEP END ################")
    return h, [h]
#######################################################
"""






class DataObject:
    def __init__(self, id, value, h_tm1_a, h_out_a, h_tm1_a_end, h_out_b, h_tm1_b_end, h_out_c):
        self.id = id
        self.value = value
        self.h_tm1_a = h_tm1_a
        self.h_out_a = h_out_a
        self.h_tm1_a_end = h_tm1_a_end
        self.h_out_b = h_out_b
        self.h_tm1_b_end = h_tm1_b_end
        self.h_out_c = h_out_c

# Initialize an empty list to store DataObject instances
data_objects = []

# Open and read the file
with open("TEST_GRU_STATE_new.log", "r") as file:
    lines = file.readlines()

# Initialize variables to store data temporarily
current_id = None
current_value = None
current_h_tm1_a = None
current_h_out_a = None
current_h_tm1_a_end = None
current_h_out_b = None
current_h_tm1_b_end = None
current_h_out_c = None
current_line_index = 0

# Iterate through the lines
for line in lines:
    line = line.strip()
    if not line:  # Skip empty lines
        current_line_index = 0
        continue
    if line.startswith("1/1 "):  # Ignore lines starting with "1/1"
        current_line_index = 0
        continue
    if line.startswith("ID: "):
        current_id = int(line[4:])
        current_line_index = 0
    elif line.startswith("VALUE: "):
        current_value = float(line[7:])
        # Create a DataObject instance and add it to the list
        data_objects.append(DataObject(current_id, current_value, current_h_tm1_a, current_h_out_a, current_h_tm1_a_end, current_h_out_b, current_h_tm1_b_end, current_h_out_c))
        # Reset the temporary variables
        current_id = None
        current_value = None
        current_h_tm1_a = None
        current_h_out_a = None
        current_h_tm1_a_end = None
        current_h_out_b = None
        current_h_tm1_b_end = None
        current_h_out_c = None
        current_line_index = 0
    else:
        if current_line_index == 1:  # 2nd valid line
            current_h_tm1_a = float(line.split("[[")[1].split("]]")[0])
        elif current_line_index == 3:  # 4th valid line
            current_h_out_a = float(line.split("[[")[1].split("]]")[0])
        elif current_line_index == 4:  # 5th valid line
            current_h_tm1_a_end = float(line.split("[[")[1].split("]]")[0])
        elif current_line_index == 5:  # 6th valid line
            current_h_out_b = float(line.split("[[")[1].split("]]")[0])
        elif current_line_index == 6:  # 7th valid line
            current_h_tm1_b_end = float(line.split("[[")[1].split("]]")[0])
        elif current_line_index == 7:  # 8th valid line
            current_h_out_c = float(line.split("[[")[1].split("]]")[0])
    current_line_index += 1

"""
# Print the DataObject instances in the list
for obj in data_objects:
    print(f"ID: {obj.id}")
    print(f"VALUE: {obj.value}")
    print(f"h_tm1_a: {obj.h_tm1_a}")
    print(f"h_out_a: {obj.h_out_a}")
    print(f"h_tm1_a_end: {obj.h_tm1_a_end}")
    print(f"h_out_b: {obj.h_out_b}")
    print(f"h_tm1_b_end: {obj.h_tm1_b_end}")
    print(f"h_out_c: {obj.h_out_c}")
    print()
"""


import matplotlib.pyplot as plt

# Extract data for the first scatter plot (h_out_a vs. h_tm1_a_end)
h_out_a_values = [obj.h_out_a for obj in data_objects]
h_tm1_a_end_values = [obj.h_tm1_a_end for obj in data_objects]

# Extract data for the second scatter plot (h_out_b vs. h_tm1_b_end)
h_out_b_values = [obj.h_out_b for obj in data_objects]
h_tm1_b_end_values = [obj.h_tm1_b_end for obj in data_objects]

# Create a figure with two subplots (side by side)
plt.figure(figsize=(16, 6))  # Adjust the figure size as needed

# First subplot (h_out_a vs. h_tm1_a_end)
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.scatter(h_out_a_values, h_tm1_a_end_values, label='h_tm1_a_end', marker='o', s=50, c='blue')
plt.xlabel('h_out_a')
plt.ylabel('h_tm1_a_end')
plt.title('Scatter Plot 1: h_out_a vs. h_tm1_a_end')
plt.grid(True)

# Second subplot (h_out_b vs. h_tm1_b_end)
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.scatter(h_out_b_values, h_tm1_b_end_values, label='h_tm1_b_end', marker='o', s=50, c='green')
plt.xlabel('h_out_b')
plt.ylabel('h_tm1_b_end')
plt.title('Scatter Plot 2: h_out_b vs. h_tm1_b_end')
plt.grid(True)

# Adjust the layout
plt.tight_layout()

# Save the figure with both scatter plots as an image file (e.g., PNG)
plt.savefig('test_gru_state.png')  # Specify the desired file name and format (e.g., 'scatter_plots.png')



data_objects.sort(key=lambda x: x.h_out_a)

with open("data_objects.txt", "w") as file:
    for obj in data_objects:
        line = f"h_out: {obj.h_out_a:.16f} | h_tm1_a_end: {obj.h_tm1_a_end}\n"
        file.write(line)
