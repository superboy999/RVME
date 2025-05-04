<!--
 * @Author: superboy
 * @Date: 2024-06-27 14:29:06
 * @LastEditTime: 2024-08-01 16:36:17
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/README.md
 * 
-->
# Attention
This repository is provided for the purpose of double-blind review. It does not contain any information identifying the authors.
Code will be released upon acceptance.
# RVME
This repo is based on gem5 simulator, to model the risc-v matrix engine.

# Change Log
- 2023/11/20
1. Modified **build_opts/RISCV**, add simulation options

- 2024/6/4
1. Support risc-v matrix extension, but just configuration, matrix-mul, and store-load.
2. Will automatically convert mb(second matrix operator) into its transpose format.
3. Compute Unit Array will compute one sub-matrix(partial sum) within one cycle.(CU Array are not connected one by one like systolic type).
4. This version will be fixed now as **main branch**.

- 2024/6/27
1. All the register operation related to the matrix memory is uint, but for the X/Y/ZBuffer they can use int or uint data. So here we will use static_cast to transfer data.

- 2024/6/28
1. Changed the CU connection to the systolic array connection.
2. Find bugs in accessing big datas, especially when access the whole cache line.
3. Add the Memchecker to the MemSystem

- 2024/7/28
1. Add SRAM Occupy Mechanism, especially for the read because write process will be protected by RAT.

-2025/2/20
1. Support new feature.
