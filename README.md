# BAD_FPGA
Bird Audio Detection with a FPGA, using one of the algorithms from the challenge (or with same dataset) Bird Audio Challenge 2017/2018.
- http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/.
- https://dcase.community/challenge2018/task-bird-audio-detection

# Acknowledgements
This project was completed as part of my Computer Science and Engeneering Master's degree at Instituto Superior de Engenharia de Lisboa, ISEL.<br>
I would like to extend my gratitude to the Professors who provided guidance and support throughout this project:
- Mário Véstias
    - https://github.com/MarioVestias
    - https://www.isel.pt/docentes/mario-pereira-vestias
- Rui Duarte
    - https://github.com/ruipduarte
    - https://www.isel.pt/docente/rui-antonio-policarpo-duarte

# Hardware
Avnet Ultra96-V2 - Xilinx Zynq UltraScale+ ZU3CG MPSoC, more information at:
- https://www.avnet.com/wps/portal/us/products/avnet-boards/avnet-board-families/ultra96-v2/
- https://www.xilinx.com/products/boards-and-kits/1-vad4rl.html
- https://docs.amd.com/v/u/en-US/zynq-ultrascale-plus-product-selection-guide

AES-ACC-U96-JTAG - JTAG adapter, more information at:
- https://www.avnet.com/shop/us/products/avnet-engineering-services/aes-acc-u96-jtag-3074457345635355958/

# Project Structure and Important Files

```
├── BAD_FPGA
|   ├── 0_quantization
│   │   ├── model_config
│   │   │   └── config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1.py
│   │   ├── cutils.py
│   │   ├── qKeras_microfaune.py
│   │   ├── qKeras_microfaune_save.py
│   │   ├── qKeras_microfaune_dump_io.py
│   │   ├── qKeras_microfaune_evaluate.py
│   │   ├── requirements.txt
│   │   ├── validate_hardware.py
│   │   └── validate_1cell.py
|   ├── 1_vitis_hls
│   │   └── mai__gC1_hcW < vitis HLS project >
|   ├── 2_vivado
│   │   ├── mai__gC1_hcW < vivado project >
│   │   └── mai__gC1_hcW.xpr.zip < archived vivado project >
|   ├── 3_vitis
│   │   └── mai__gC1_hcW < vitis project >
|   ├── 4_demo
│   │   └── demo.py
|   ├── 5_project_report
│   ├── python-algorithms
│   │   ├── All-Conv-...  < submodule >
│   │   └── microfaune_ai < submodule >
│   └── seed_finder
├── files_audio
│   ├── ff1010bird
│   │   └── wav < folder containing WAV audio files >
│   ├── warblrb10k_evaluate
│   │   └── wav < folder containing WAV audio files >
│   └── warblrb10k_public
│   │   └── wav < folder containing WAV audio files >
└── files_audio_metadata
    ├── ff1010bird_metadata.csv
    └── warblrb10k_public_metadata.csv
```
The folder structure shows 1 folder level above, ```BAD_FPGA``` being this repository.<br>
The folders ```files_audio``` and ```files_audio_metada``` are the datasets folders, taken from the DCASE community website, Challenge2018 Task3.

This repository have the files used during development and tests made along the way, but the important files can be seen in the Project structure shown.

All the scripts start with a variables section, and some might require a change to the base path, dataset location or model location.
If during execution it is reported a missing file, start by checking those path variables at the top of the script file.

# How to make it work
The Operative System used was Windows 10, but the python scripts were executed under Windows Subsystem for Linux, with Ubuntu 22.04.2 LTS installed.

## 0_quantization
The first folder and the start of the project.

First you might need to get the used library versions, which are present in the file ```requirements.txt```. These were the installed libraries at the end of this project.

The next scripts that start with ```qKeras_microfaune``` are the scripts to train and quantize, weights extraction, output extraction and evaluation.<br>
These scripts have an important thing in commun, at the top of the script code, each script have the QKeras model configuration used.<br>
This is defined using an ```import``` easily seen by the amount of model configurations tested.<br>
This configuration must be the same across these next scripts, and at the time of this commit this configuration is:<br>
```python
from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1 import ModelConfig
```

### Training and Auantization
Next, and after downloading the datasets and placing them accordingly to the __Project Structure__, the first python script to be executed is:<br>
>```qKeras_microfaune.py``` - Model training and QKeras quantization

This will start training the microfaune_ai model and use QKeras quantization.

### Weights Extraction
Then, to extract the trained model weights the next script is used.
>```qKeras_microfaune_save.py``` - Model weights extraction after training

### Output Extraction for a Single Input (Optional)
This step is optional, but if you want compare the outputs of this trained QKeras model with teh output of the FPGA, this might be useful.
>```qKeras_microfaune_dump_io.py``` - Model outputs extraction

### QKeras Evaluation (Optional)
This step is also optional since it is used to evaluate an input using the QKeras quantized model.
>```qKeras_microfaune_evaluate.py``` - Model evaluation using an audio file

### Creating a Binary File for FPGA Evaluation
both of these next 2 scripts have a variable ```nSelByClass```, that is a value that tells how many of inputs for each classification should be selected.
If set to 200, means that 200 are negative classification and 200 for positive, totalling 400 inputs.
If you change this, remember to change the ```#define INPUT_SIZE``` in ```3_vitis/mai__gC1_hcW/microfaune_ai/src/microfaune_ai.cpp``` to 2 times the value you set on the python script.<br>
Example, you set the ```nSelByClass = 400```, then ```#define INPUT_SIZE 800```.

#### validate_hardware.py
This script creates a binary with multiple inputs concatenated, ready for FPGA use.
This binary file is then used with the ```3_vitis/mai_gC1_hcW project``` when programming the FPGA.
This binary file is divided in half, with the 1st half being only negative inputs (no bird) and the 2nd half only positive inputs (have bird).

#### validate_1cell.py
Can be used to check the accucary of the original microfaune model with only 1 GRU cell for thise same sample size, for easy comparison with the hardware implementation.

## 1_vitis_hls/mai__gC1_hcW
Now is time to create the Hardware IP Blocks, with the ```1_vitis_hls/mai__gC1_hcW``` project, in Vitis HLS.

The IP Blocks cannot be synthesised at once.
For that reason, the ```top function``` under ```Project Settings``` must be set to:
- __conv2D__ - For the Convolution Hardware IP Block
- __gru__ - For the Bidirectional GRU Hardware IP Block

Export them for later use in Vivado.

## 2_vivado/mai__gC1_hcW
Open the ```2_vivado/mai__gC1_hcW``` project and update the IP Blocks with the previously exported IP Blocks.

Let it run, and then to to Vitis.

## 3_vitis/mai__gC1_hcW
In Vitis HLS, open the ```3_vitis/mai__gC1_hcW``` project and use the design wrapper created in Vivado.<br>
Note: The ```#define INPUT_SIZE``` explained earlier should be updated at this stage. The binary file with those packed inputs, should also be added at this stage to the Run Configuration under ```Run Configurations > Application > Advanced Options > Edit```. Here add the binary file at the address ```0x1000000```.

Now, compile and execute, it will program the FPGA and then you will see the result on the Vitis IDE console.
Note that for large binary files the programming process might take a long time, since the binary file is being sent over UART. With a binary file of 400 inputs it can take up to 15 minutes.

## 4_demo (Optional)
To run the demo used in the Power Point presentation, the process is the same until ```3_vitis/mai__gC1_hcW```.
Also, ```validate_hardware.py``` python script is not required because the demo does not use that binary file, but WAV files instead.<br>
In place of the ```3_vitis/mai__gC1_hcW```, you must program the FPGA using the ```4_demo/vitis``` Vitis project.<br>

If you are using Windows WSL, that Linux will not see the USB device.
To solve this issue, you can use ```usbipd-win``` found in the repository: https://github.com/dorssel/usbipd-win.<br>
You can also see the commands i used to connect and disconnect the USB inside the file ```usb2wsl.txt```.
Note that every time you have to reprogram the FPGA, you need to detach from the WSL.

Now, just place some WAV files from the datasets into the ```4_demo/audio_files``` folder and run the ```demo.py``` script.<br>
If you have problems with demo script selecting the incorrect USB port, the variable ```ports``` is an array of all the USB devices connected to the WSL, just change the index used to affect the variable ```port``` to the correct one.

## 5_project_report
This folder contains the final Thesis report, the Thesis summary, and the Power point presentation.