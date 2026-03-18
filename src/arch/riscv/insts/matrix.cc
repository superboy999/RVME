/*
 * @Author: superboy 
 * @Date: 2025-02-27 10:03:16 
 * @Last Modified by: super_squirrel
 * @Last Modified time: 2025-02-27 18:49:58
 */

// #include "arch/riscv/insts/mattor.hh"

#include <sstream>
#include <string>

#include "arch/riscv/insts/static_inst.hh"
#include "arch/riscv/insts/matrix_static_inst.hh"
#include "arch/riscv/isa.hh"
#include "arch/riscv/regs/misc.hh"
#include "cpu/exec_context.hh"
// #include "arch/riscv/regs/mattor.hh"
#include "arch/riscv/utility.hh"
#include "cpu/static_inst.hh"
#include "arch/riscv/regs/int.hh"
#include "arch/riscv/insts/matrix.hh"
#include "arch/riscv/regs/matrix.hh"

namespace gem5
{

namespace RiscvISA
{

    std::string
    RiscvMatrixConfIOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
    {
        std::stringstream ss;
        ss << mnemonic << ' ' << uimm;
        return ss.str();
    }

    std::string
    RiscvMatrixConfOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
    {
        std::stringstream ss;
        ss << mnemonic << ' ' << int_reg::RegNames[mrs1()];
        return ss.str();
    }

    std::string
    RiscvMatrixMultiplyOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
    {
        std::stringstream ss;
        ss << mnemonic << ' ' << matrix_reg::RegNames[md()] << ", " << matrix_reg::RegNames[ms2()] << ", " << matrix_reg ::RegNames[ms1()];
        return ss.str();
    }
    
    std::string
    RiscvMatrixMoveOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
    {
        std::stringstream ss;
        if (mfunct4() == 2) {
            ss << mnemonic << ' ' << int_reg::RegNames[mrd()] << ", " << matrix_reg::RegNames[ms2()] << ", " << int_reg::RegNames[mrs1()];
        }
        else if (mfunct4() == 3) {
            ss << mnemonic << ' ' << matrix_reg::RegNames[md()] << ", " << int_reg::RegNames[mrs1()] << ", " << int_reg::RegNames[mrs2()];
        }
        else {
            ss << mnemonic << ' ' << matrix_reg::RegNames[md()] << ", " << matrix_reg::RegNames[ms1()];
        }
        return ss.str();
    }
    
    std::string
    RiscvMatrixElementwiseOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
    {
        std::stringstream ss;
        ss << mnemonic << ' ' << matrix_reg::RegNames[md()] << ", " << matrix_reg::RegNames[ms2()] << ", " << matrix_reg ::RegNames[ms1()];
        return ss.str();
    }
    
    std::string
    RiscvMatrixLoadOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
    {
        std::stringstream ss;
        ss << mnemonic << ' ' << matrix_reg::RegNames[md()] << ", (" << int_reg::RegNames[mrs1()] << "), " << int_reg::RegNames[mrs2()];
        return ss.str();
    }

    std::string
    RiscvMatrixStoreOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
    {
        std::stringstream ss;
        ss << mnemonic << ' ' << matrix_reg::RegNames[ms3()] << ", (" << int_reg::RegNames[mrs1()] << "), " << int_reg::RegNames[mrs2()];
        return ss.str();
    }

    std::string
    RiscvMatrixZeroOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
    {
        std::stringstream ss;
        ss << mnemonic << ' ' << matrix_reg::RegNames[md()];
        return ss.str();
    }

    std::string
    RiscvMatrixLutOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
    {
        std::stringstream ss;
        ss << mnemonic << ' ' << matrix_reg::RegNames[md()] << ", " << int_reg::RegNames[ms1()];
        return ss.str();
    }
    
    std::string
    RiscvMatrixRedcOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
    {
        std::stringstream ss;
        ss << mnemonic << ' ' << matrix_reg::RegNames[md()] << ", " << int_reg::RegNames[ms1()];
        return ss.str();
    }
    
    std::string
    RiscvMatrixSyncOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
    {
        std::stringstream ss;
        ss << mnemonic;
        return ss.str();
    }

}

}