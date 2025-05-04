// ==========================================
// Author: cwq
// Last Date: 2023/11/28
// Description: Build matrix register program
// -ming model here
// ==========================================

#ifndef __ARCH_RISCV_REGS_MATRIX_HH__
#define __ARCH_RISCV_REGS_MATRIX_HH__

#include <specialize.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cpu/reg_class.hh"
#include "debug/MatrixRegs.hh"

namespace gem5
{

namespace RiscvISA
{

namespace matrix_reg
{

enum : RegIndex
{
    _M0Idx, _M1Idx, _M2Idx, _M3Idx,
    _M4Idx, _M5Idx, _M6Idx, _M7Idx,

    NumRegs
};

} // matrix_reg

inline constexpr RegClass matrixRegClass(MatrixRegClass, MatrixRegClassName,
        matrix_reg::NumRegs, debug::MatrixRegs);

namespace matrix_reg
{

inline constexpr RegId
    M0 = matrixRegClass[_M0Idx],
    M1 = matrixRegClass[_M1Idx],
    M2 = matrixRegClass[_M2Idx],
    M3 = matrixRegClass[_M3Idx],
    M4 = matrixRegClass[_M4Idx],
    M5 = matrixRegClass[_M5Idx],
    M6 = matrixRegClass[_M6Idx],
    M7 = matrixRegClass[_M7Idx];    

const std::vector<std::string> RegNames = {
    "m0", "m1", "m2", "m3",
    "m4", "m5", "m6", "m7"
};
}

} // RiscvISA
} // gem5


#endif // __ARCH_RISCV_REGS_MATRIX_HH__