import subprocess as sp
import tensorflow as tf

def gpu_memory_usage(gpu_id):
    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.used --format=csv"
    output_cmd = sp.check_output(command.split())
    
    memory_used = output_cmd.decode("ascii").split("\n")[1]
    # Get only the memory part as the result comes as '10 MiB'
    memory_used = int(memory_used.split()[0])

    return memory_used

# The gpu you want to check
gpu_id = 0

initial_memory_usage = gpu_memory_usage(gpu_id)

# Set up the gpu specified
gpu_physical_devices = tf.config.list_physical_devices('GPU')
for device in gpu_physical_devices:
    if int(device.name.split(":")[-1]) == gpu_id:
        device_to_be_used = device
        # Set memory growth for TF to not use all available memory of the GPU
        tf.config.experimental.set_memory_growth(device, True)

# Just to be sure that we are only using the required gpu
tf.config.set_visible_devices([device_to_be_used], 'GPU')


# Create your model here
# Do cool stuff ....

latest_gpu_memory = gpu_memory_usage(gpu_id)
print(f"(GPU) Memory used: {latest_gpu_memory - initial_memory_usage} MiB")