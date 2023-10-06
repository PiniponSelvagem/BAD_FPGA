
##################################################
# TAKEN FROM test_gry_state_qkeras__READ.py

fileName = "TEST_GRU_STATE_3_stateQuant_randomWeights"

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
with open(f"{fileName}.log", "r") as file:
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

##################################################






# Extract data for the first scatter plot (h_out_a vs. h_tm1_a_end)
h_out_a_values = [obj.h_out_a for obj in data_objects]
h_tm1_a_end_values = [obj.h_tm1_a_end for obj in data_objects]

# Extract data for the second scatter plot (h_out_b vs. h_tm1_b_end)
h_out_b_values = [obj.h_out_b for obj in data_objects]
h_tm1_b_end_values = [obj.h_tm1_b_end for obj in data_objects]

data_objects.sort(key=lambda x: x.h_out_a)



def stateQuant(state):
    OFFSET_STATE = 0.125
    STEP_SIZE_STATE = 0.25
    MIN_STATE = -1
    MAX_STATE = 1

    state = state + OFFSET_STATE
    if (state <= MIN_STATE):
        return MIN_STATE
    elif (state >= MAX_STATE):
        return MAX_STATE
    step = int((state + 1) / STEP_SIZE_STATE)
    return (step * STEP_SIZE_STATE) - 1.0



all_match = True
for obj in data_objects:
    calculated_value = stateQuant(obj.h_out_a)
    
    if calculated_value != obj.h_tm1_a_end:
        print(f"h_out_a: {obj.h_out_a:.16f}, h_tm1_a_end: {obj.h_tm1_a_end}, stateQuant: {calculated_value:.16f}")
        all_match = False

if all_match:
    print("Success")

