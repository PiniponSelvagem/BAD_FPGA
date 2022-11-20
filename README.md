# BAD_FPGA
Bird Audio Detection with a FPGA, using one of the algorithms from the challenge (or with same dataset) Bird Audio Challenge 2017/2018.
- http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/.

# Hardware
ArduZynq - Xilinx Zynq-7010, more information at:
- https://shop.trenz-electronic.de/en/TE0723-03-41C64-A-ArduZynq-Arduino-compatible-Xilinx-Zynq-7010-SoC-Modul?c=319

# Project structure
- it is suggested to place audio files and their metadata 1 level above with the following structure:
```
├── BAD_FPGA
│   ├── python-algorithms
│   │   ├── All-Conv-...  < submodule >
│   │   └── microfaune_ai < submodule >
│   └── python-quantization
│       └── < algorithms quantization >
├── files_audio
│   ├── ff1010bird
│   ├── warblrb10k_evaluate
│   └── warblrb10k_public
└── files_audio_metadata
    ├── ff1010bird_metadata.csv
    └── warblrb10k_public_metadata.csv
```


