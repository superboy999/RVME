/*
 * @Author: superboy 
 * @Date: 2025-02-27 10:02:44 
 * @Last Modified by: super_squirrel
 * @Last Modified time: 2025-02-27 11:20:10
 */

#ifndef __ARCH_RISCV_INSTS_MATRIX_HH__
#define __ARCH_RISCV_INSTS_MATRIX_HH__

#include <string>

#include "arch/riscv/insts/static_inst.hh"
#include "arch/riscv/insts/static_inst.hh"
#include "arch/riscv/isa.hh"
#include "arch/riscv/regs/misc.hh"
#include "cpu/exec_context.hh"
#include "arch/riscv/insts/matrix_static_inst.hh"
#include "arch/riscv/regs/int.hh"

namespace gem5
{
namespace RiscvISA
{
    /* matrix configuration instructions */
    class RiscvMatrixConfIOp : public RiscvMatrixInst
    {
        protected:
            uint64_t uimm;
        public:
        RiscvMatrixConfIOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass), uimm(_machInst.mimm10)
        {}

        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };  

    class RiscvMatrixConfOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixConfOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {}

        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };  
  
    /* matrix arithmetic Instructions*/
    class RiscvMatrixMultiplyOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixMultiplyOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {}

        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };

    class RiscvMatrixMoveOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixMoveOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {}

        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };

    class RiscvMatrixElementwiseOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixElementwiseOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {}

        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };

    /* matrix memory Instructions*/
    class RiscvMatrixLoadOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixLoadOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {}
        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };

    class RiscvMatrixZeroOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixZeroOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {}
        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };

    class RiscvMatrixLutOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixLutOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {}

        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };
    
    class RiscvMatrixRedcOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixRedcOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {}

        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };

    class RiscvMatrixStoreOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixStoreOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {}
        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };

    class RiscvMatrixSyncOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixSyncOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {}
        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };

} // namespace RiscvISA
}  // namespace gem5




#endif // __ARCH_RISCV_INSTS_MATRIX_HH__
