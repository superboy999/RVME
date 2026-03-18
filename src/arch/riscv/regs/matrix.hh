/*
 * @Author: superboy
 * @Date: 2025-03-05 20:27:50
 * @LastEditTime: 2025-03-07 20:28:28
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /SJTU-matrix-engine/src/arch/riscv/regs/matrix.hh
 * 
 */

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
    _TR0Idx, _TR1Idx, _TR2Idx, _TR3Idx,
    _ACC0Idx, _ACC1Idx, _ACC2Idx, _ACC3Idx,

    NumRegs
};

} // matrix_reg

inline constexpr RegClass matrixRegClass(MatrixRegClass, MatrixRegClassName,
        matrix_reg::NumRegs, debug::MatrixRegs);

namespace matrix_reg
{

inline constexpr RegId
    TR0 = matrixRegClass[_TR0Idx],
    TR1 = matrixRegClass[_TR1Idx],
    TR2 = matrixRegClass[_TR2Idx],
    TR3 = matrixRegClass[_TR3Idx],
    ACC0 = matrixRegClass[_ACC0Idx],
    ACC1 = matrixRegClass[_ACC1Idx],
    ACC2 = matrixRegClass[_ACC2Idx],
    ACC3 = matrixRegClass[_ACC3Idx];    

const std::vector<std::string> RegNames = {
    "tr0", "tr1", "tr2", "tr3",
    "acc0", "acc1", "acc2", "acc3"
};
}

} // RiscvISA
} // gem5


#endif // __ARCH_RISCV_REGS_MATRIX_HH__