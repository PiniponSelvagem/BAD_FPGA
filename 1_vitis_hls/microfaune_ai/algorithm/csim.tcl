############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
## Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
############################################################
open_project microfaune_ai
add_files microfaune_ai/source/bnorm/bnorm.h
add_files microfaune_ai/source/conv2d/conv2d.h
add_files microfaune_ai/source/load_weights.h
add_files microfaune_ai/source/predict.cpp
add_files -tb microfaune_ai/test_bench/main.cpp -cflags "-Wno-unknown-pragmas"
open_solution "algorithm" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
#source "./microfaune_ai/algorithm/directives.tcl"
csim_design -quiet
