############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
############################################################
open_project cnn_rnn
set_top predict
add_files cnn_rnn/cnn_rnn/source/settings.h
add_files cnn_rnn/cnn_rnn/source/cnn_rnn.cpp
add_files -tb cnn_rnn/cnn_rnn/test_bench/data.h
add_files -tb cnn_rnn/cnn_rnn/test_bench/tb_cnn_rnn.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "cnn_rnn" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
config_export -format ip_catalog -rtl verilog -vivado_clock 10
#source "./cnn_rnn/cnn_rnn/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -flow syn -rtl verilog -format ip_catalog
