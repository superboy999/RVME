# GEM5-RVME
This repository is based on the gem5 simulator and provides a detailed microarchitectural model of **RVME**, as described in our paper accepted by ICCD 2025, titled “**RVME: An Efficient Matrix Engine Design Based on Matrix Extension of RISC-V.**”


Now RVME supports the main instructions metioned in [toolchain](toolchain/).
## How to use
1. Build environment, pull gem5 docker.
``` shell
docker pull ghcr.io/gem5/ubuntu-24.04_all-dependencies:build-cache
```
2. Compile RVME model.
``` shell
scons build/RISCV/gem5.debug -j 80  // .opt will also be fine, even faster. 80 can be changed to actual hardware thread you can use.
```
3. Runing test on RVME in debugging mode.(By GDB)
``` shell
gdb build/RISCV/gem5.debug
run --debug-flags=<add some flags of modules that you want to debug> configs/example/riscv_matrix_engine.py 
```