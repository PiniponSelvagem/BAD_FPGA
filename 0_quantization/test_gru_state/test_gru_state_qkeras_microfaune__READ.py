import numpy as np

b = 4

# Open and read the file
with open(f"TEST_GRU_STATE_quant{b}01.log", "r") as file:
    lines = file.readlines()

# Initialize variables to store data temporarily
h_out = np.array([])
h_tm1_end = np.array([])

# Iterate through the lines
for line in lines:
    line = line.strip()
    data_string = line.split("[[")[1].split("]]")[0]
    data = np.fromstring(data_string, sep=" ")
    #
    if line.startswith("h ="):
        h_out = np.append(h_out, data)
    else:  # starts with "h_tm1"
        h_tm1_end = np.append(h_tm1_end, data)


for i in range(0, 2000):
    out = h_out[i]
    state = h_tm1_end[i]
    if (out+0.0625 < state or out-0.0625 > state):
        print(i, out, state)

#868
#955



import matplotlib.pyplot as plt
plt.clf()

plt.figure(figsize=(18, 8))
plt.scatter(h_out, h_tm1_end, label='h_tm1_end', marker='o', s=1, c='blue')
plt.xlabel('h_out')
plt.ylabel('h_tm1_end')
plt.title('Scatter Plot 1: h_out vs. h_tm1_end')
plt.grid(False)

custom_xticks = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
plt.xticks(custom_xticks, custom_xticks)
custom_yticks = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
plt.yticks(custom_yticks, custom_yticks)

plt.savefig(f'test_gru_state_microfaune_{b}.png')


"""
scaled = 10
downsampled_h_out = h_out[::scaled]
downsampled_h_tm1_end = h_tm1_end[::scaled]
plt.clf()

plt.figure(figsize=(18, 8))
plt.scatter(downsampled_h_out, downsampled_h_tm1_end, label='downsampled_h_tm1_end', marker='o', s=1, c='blue')
plt.xlabel('downsampled_h_out')
plt.ylabel('downsampled_h_tm1_end')
plt.title('Scatter Plot 1: downsampled_h_out vs. downsampled_h_tm1_end')
plt.grid(False)

custom_xticks = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
plt.xticks(custom_xticks, custom_xticks)
custom_yticks = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
plt.yticks(custom_yticks, custom_yticks)


plt.savefig(f'test_gru_state_microfaune_{b}_DOWNSAMPLED.png')
"""