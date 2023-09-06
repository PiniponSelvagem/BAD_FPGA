############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
## Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
############################################################
open_project cnn_rnn
set_top predict
add_files cnn_rnn/cnn_rnn/source/predict.cpp
add_files -tb cnn_rnn/cnn_rnn/test_bench/main.cpp -cflags "-Wno-unknown-pragmas"
open_solution "cnn_rnn" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
source "./cnn_rnn/cnn_rnn/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
