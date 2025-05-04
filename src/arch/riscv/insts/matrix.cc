// ==========================================
// Author: cwq
// Last Date: 2023/11/27
// ==========================================
#include <string>
#include <sstream>

#include "arch/riscv/insts/static_inst.hh"
#include "arch/riscv/regs/misc.hh"
#include "cpu/exec_context.hh"
#include "cpu/static_inst.hh"
#include "arch/riscv/insts/matrix_static_inst.hh"
#include "arch/riscv/regs/int.hh"
#include "arch/riscv/insts/matrix.hh"
#include "arch/riscv/utility.hh"

namespace gem5
{

namespace RiscvISA
{

std::string
RiscvMatrixMultiplyOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
{
    std::stringstream ss;
    ss << mnemonic << ' ' << matrix_reg::RegNames[md()] << ", " << matrix_reg::RegNames[ms2()] << ", " << matrix_reg ::RegNames[ms1()];
    return ss.str();
}

std::string
RiscvMatrixCfgOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
{
    std::stringstream ss;
    ss << mnemonic << ' ' << registerName(destRegIdx(0)) << ", "<< registerName(srcRegIdx(0));
    return ss.str();
}

std::string
RiscvMatrixCfgIOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
{
    std::stringstream ss;
    ss << mnemonic << ' ' << registerName(destRegIdx(0)) << ", " << uimm;
    return ss.str();
}


std::string
RiscvMatrixMemOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
{
    std::stringstream ss;
    ss << mnemonic << ' ' << matrix_reg::RegNames[md()] << ", " <<registerName(srcRegIdx(1)) << ", " << "(" << registerName(srcRegIdx(0)) << ")";
    return ss.str();
}

} // namespace RiscvISA
} // namespace gem5 