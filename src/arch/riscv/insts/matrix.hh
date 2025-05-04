/*
 * Copyright (c) 2015 RISC-V Foundation
 * Copyright (c) 2017 The University of Virginia
 * Copyright (c) 2020 Barkhausen Institut
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// ==========================================
// Author: cwq
// Last Date: 2023/11/20
// ==========================================
#ifndef __ARCH_RISCV_MATRIX_INST_HH__
#define __ARCH_RISCV_MATRIX_INST_HH__

#include <string>

#include "arch/riscv/insts/static_inst.hh"
#include "arch/riscv/regs/misc.hh"
#include "cpu/exec_context.hh"
#include "base/bitfield.hh"
// #include "cpu/static_inst.hh"
#include "arch/riscv/insts/matrix_static_inst.hh"
#include "arch/riscv/regs/int.hh"

namespace gem5
{

namespace RiscvISA
{
    /* matrix Arithmetic Instructions*/
    class RiscvMatrixMultiplyOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixMultiplyOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {
            // _numSrcRegs = 0;
            // _numDestRegs = 0;
        }

        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };

    class RiscvMatrixCfgOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixCfgOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {
            // _numSrcRegs = 1;
            // _numDestRegs = 1;
            // setSrcRegIdx(0, intRegClass[mrs1()]);
            // setDestRegIdx(0, intRegClass[mrd()]);
        }

        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };

    class RiscvMatrixCfgIOp : public RiscvMatrixInst
    {
        protected:
            uint64_t uimm;
        public:
        RiscvMatrixCfgIOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass), uimm(_machInst.muimm7)
        {
            // considering about all the cfg inst are mcfgm/n/k, they just need one scalar register
            // _numSrcRegs = 0;
            // _numDestRegs = 1;
            // setDestRegIdx(0, intRegClass[mrd()]);
        }
        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };

    class RiscvMatrixMemOp : public RiscvMatrixInst
    {
        public:
        RiscvMatrixMemOp(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass):
        RiscvMatrixInst(_mnemonic, _machInst, __opClass)
        {
            // _numSrcRegs = 2;
            // _numDestRegs = 0;
            // setSrcRegIdx(0, intRegClass[mrs1()]);
            // setSrcRegIdx(1, intRegClass[mrs2()]);
        }
        std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const override;
    };

} // namespace RiscvISA
} // namespace gem5

#endif // __ARCH_RISCV_MATRIX_INST_HH__
