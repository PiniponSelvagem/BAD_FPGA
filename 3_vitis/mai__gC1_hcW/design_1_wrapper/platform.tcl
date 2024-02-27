# 
# Usage: To re-create this platform project launch xsct with below options.
# xsct D:\BAD_FPGA\3_vitis\mai__gC1_hcW\design_1_wrapper\platform.tcl
# 
# OR launch xsct and run below command.
# source D:\BAD_FPGA\3_vitis\mai__gC1_hcW\design_1_wrapper\platform.tcl
# 
# To create the platform in a different location, modify the -out option of "platform create" command.
# -out option specifies the output directory of the platform project.

platform create -name {design_1_wrapper}\
-hw {D:\BAD_FPGA\2_vivado\mai__gC1_hcW\design_1_wrapper.xsa}\
-arch {64-bit} -fsbl-target {psu_cortexa53_0} -out {D:/BAD_FPGA/3_vitis/mai__gC1_hcW}

platform write
domain create -name {standalone_psu_cortexa53_0} -display-name {standalone_psu_cortexa53_0} -os {standalone} -proc {psu_cortexa53_0} -runtime {cpp} -arch {64-bit} -support-app {empty_application}
platform generate -domains 
platform active {design_1_wrapper}
domain active {zynqmp_fsbl}
domain active {zynqmp_pmufw}
domain active {standalone_psu_cortexa53_0}
platform generate -quick
bsp reload
bsp config stdin "psu_uart_1"
bsp config stdout "psu_uart_1"
bsp write
bsp reload
catch {bsp regenerate}
platform config -updatehw {D:/BAD_FPGA/2_vivado/mai__gC1_hcW/design_1_wrapper.xsa}
platform generate
