############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
## Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
############################################################
open_project microfaune_ai
set_top predict
add_files microfaune_ai/source/axis_bgru.cpp
add_files microfaune_ai/source/axis_conv3D.cpp
add_files microfaune_ai/source/global_settings.h
add_files microfaune_ai/source/size_bgru.h
add_files microfaune_ai/source/size_conv3D.h
add_files microfaune_ai/source/types.h
add_files microfaune_ai/source/utils.h
add_files -tb microfaune_ai/source/loader.h
add_files -tb microfaune_ai/source/soft_reducemax.h -cflags "-Wno-unknown-pragmas"
add_files -tb microfaune_ai/source/soft_timedist.h -cflags "-Wno-unknown-pragmas"
add_files -tb microfaune_ai/source/tb_main.cpp -cflags "-Wno-unknown-pragmas"
open_solution "algorithm" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
source "./microfaune_ai/algorithm/directives.tcl"
csim_design -clean
csynth_design
cosim_design
export_design -format ip_catalog
