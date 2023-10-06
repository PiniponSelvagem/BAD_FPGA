############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
## Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
############################################################
open_project microfaune_ai
set_top conv2d
add_files bnorm.h
add_files microfaune_ai/source/conv2d/conv2d.h
add_files microfaune_ai/source/gru/gru.h
add_files microfaune_ai/source/load_weights.h
add_files microfaune_ai/source/mpool2d/maxpool2d.h
add_files microfaune_ai/source/predict.cpp
add_files microfaune_ai/source/reducemax/reducemax.h
add_files microfaune_ai/source/timedist/timedist.h
add_files -tb microfaune_ai/test_bench/main.cpp -cflags "-Wno-unknown-pragmas"
open_solution "algorithm" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
source "./microfaune_ai/algorithm/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
