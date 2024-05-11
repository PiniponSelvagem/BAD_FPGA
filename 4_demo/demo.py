import os
import time
from datetime import timedelta

import numpy as np
import serial

import audio2input


##################################################

folder_audio = "audio_files"


files = os.listdir(folder_audio)

print("Files in the folder:")
for i, file in enumerate(files):
    print(f"{i}. {file}")

selection = input("Enter the number of the file you want to select: ")

try:
    selection_index = int(selection)
    selected_file = files[selection_index]
    print(f"Selected file: {selected_file}")
    full_path = os.path.join(folder_audio, selected_file)
    print(f"Full path of the selected file: {full_path}")
except (ValueError, IndexError):
    print("Invalid selection. Please enter a valid number.")



bin_data = audio2input.getHardwareInput(full_path)
input_data = bin_data.tobytes()



import sys
import glob
import serial


def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')
    #
    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


print(serial_ports())


port = serial_ports()[0]
print(port)
ser = serial.Serial(port, 115200)

import time

try:
    while True:
        ser.write(input_data)
        time.sleep(5)
        # Read bytes from the serial port
        data = ser.read_all()  # Read up to 100 bytes
        # Print the received bytes
        print("Received:", data)
        # Do something with the received data
        time.sleep(5)
except KeyboardInterrupt:
    print("Keyboard interrupt. Exiting...")

ser.close()
