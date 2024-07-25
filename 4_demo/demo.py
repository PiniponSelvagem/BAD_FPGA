import os
import time

import sys
import glob
import serial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import audio2input

##################################################

folder_audio = "audio_files"
vlc_path = "/mnt/c/Program Files/VideoLAN/VLC/vlc.exe"

##################################################
ascii_bird = """                 _.----._ 
               ,'.::.--..:._
              /::/_,-<o)::;_`-._
             ::::::::`-';'`,--`-`
             ;::;'|::::,','
           ,'::/  ;:::/, :.
          /,':/  /::;' \ ':\\
         :'.:: ,-''   . `.::\\
         \.:;':.    `    :: .:
         (;' ;;;       .::' :|
          \,:;;      \ `::.\.\\
          `);'        '::'  `:
           \.  `        `'  .:      _,'
            `.: ..  -. ' :. :/  _.-' _.-
              >;._.:._.;,-=_(.-'  __ `._
            ,;'  _..-((((''  .,-''  `-._
         _,'<.-''  _..``'.'`-'`.        `
     _.-((((_..--''       \ \ `.`.
   -'  _.``'               \      ` SSt
     ,'
"""
#Thank you for visiting https://asciiart.website/
#This ASCII pic can be found at
#https://asciiart.website/index.php?art=animals/birds%20(land)

ascii_nobird = """  _  _    ___  _  _         _     _         _               _      __                       _ 
 | || |  / _ \| || |    _  | |__ (_)_ __ __| |  _ __   ___ | |_   / _| ___  _   _ _ __   __| |
 | || |_| | | | || |_  (_) | '_ \| | '__/ _` | | '_ \ / _ \| __| | |_ / _ \| | | | '_ \ / _` |
 |__   _| |_| |__   _|  _  | |_) | | | | (_| | | | | | (_) | |_  |  _| (_) | |_| | | | | (_| |
    |_|  \___/   |_|   (_) |_.__/|_|_|  \__,_| |_| |_|\___/ \__| |_|  \___/ \__,_|_| |_|\__,_|
"""
#Created at:
#https://patorjk.com/software/taag/#p=display&f=Ivrit&t=404%20%3A%20bird%20not%20found
##################################################

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

ports = serial_ports()
print("INFO: USB devices found", ports)

port = ports[0]
baudrate = 115200
print(f"INFO: Selecting port '{port}' with {baudrate}")
ser = serial.Serial(port, baudrate)


try:
    while True:
        print("################################################################")
        files = os.listdir(folder_audio)

        print("Folder:", folder_audio)
        for i, file in enumerate(files):
            print(f"{i}. {file}")

        option_notvalid = True
        while (option_notvalid):
            selection = input("Option: ")
            try:
                selection_index = int(selection)
                selected_file = files[selection_index]
                full_path = os.path.join(folder_audio, selected_file)
                print(f"Selected audio: {selected_file}")
                option_notvalid = False
            except (ValueError, IndexError):
                print("Invalid option. Please enter a valid number.")

        import subprocess
        def play_audio_vlc(file_path):
            command = [vlc_path , '--play-and-exit', file_path]
            subprocess.run(command)
        print("Playing audio with VLC")
        play_audio_vlc(full_path)
        
        print("Creating feature map... ")
        bin_data = audio2input.getHardwareInput(full_path)
        input_data = bin_data.tobytes()
        print("> DONE")

        print("Sending feature map... ")
        ser.write(input_data)
        ser.flush()
        print("> DONE")

        while ser.in_waiting == 0:
            pass

        print("Receiving... ")
        data = ser.read_all()
        print("> DONE")
        percentage = float(data)
        if (percentage >= 0.50):
            ascii_art = ascii_bird
        else:
            ascii_art = ascii_nobird
        print(ascii_art)
        print(f"Result = {percentage} ({str(round(float(data)*100, 2))} %)")
        time.sleep(5)
except KeyboardInterrupt:
    print("Keyboard interrupt. Exiting...")

ser.close()
