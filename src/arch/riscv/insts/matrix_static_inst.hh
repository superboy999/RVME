// ==========================================
// Author: cwq
// Last Date: 2023/11/20
// ==========================================
#ifndef __ARCH_RISCV_MATRIX_STATIC_INST_HH__
#define __ARCH_RISCV_MATRIX_STATIC_INST_HH__

#include <string>

#include "arch/riscv/insts/static_inst.hh"
#include "arch/riscv/pcstate.hh"
#include "arch/riscv/types.hh"
#include "cpu/exec_context.hh"
// #include "cpu/static_inst.hh"
#include "cpu/thread_context.hh"
#include "mem/packet.hh"
#include "arch/riscv/regs/misc.hh"
#include "base/types.hh"

namespace gem5
{

namespace RiscvISA
{

    class RiscvMatrixInst : public RiscvStaticInst
    {
    protected:
        // using RiscvStaticInst::RiscvStaticInst;
        RiscvMatrixInst(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass) :
        RiscvStaticInst(_mnemonic, _machInst, __opClass),
        b(_machInst), mnemonic(_mnemonic){}
        ~RiscvMatrixInst(){}

    public:
        RegIndex ms1() const { return x(18, 3);}
        RegIndex ms2() const { return x(21, 3);}
        RegIndex ms3() const {return x(7, 3);}
        RegIndex md() const { return x(15, 3);}
        RegIndex load_md() const { return x(7, 3);}
        RegIndex mrs1() const { return x(15, 5);}
        RegIndex mrs2() const { return x(20, 5);}
        RegIndex mrd() const { return x(7, 5);}
        uint8_t uimm7() const {return x(18, 7);}
        uint8_t getbits_0_7() const {return x(0, 8);}
        uint8_t getbits_8_15() const {return x(8, 8);}
        uint16_t getbits_16_31() const {return x(16, 16);}
        uint8_t getbits_31() const {return x(31, 1);}
        uint8_t getbits_18_24() const {return x(18, 7);}
    public:
        void advancePC(PCStateBase &pc) const override
        {
            pc.as<PCState>().advance();
        }
        std::string getName() const
        {
            return mnemonic;
        }
        virtual std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const=0;

        bool isMatrixInst() const {
            return x(0, 7) == 0b0101011;
           // return b.m_majorcode == 0b;
        }
        bool isMatrixConfig() const {
            return x(25, 3) == 0b111;
            //return b.muop == 0b111
        }
        // virtual bool isMatrixInstMem() const {
        //     return x(               )
        // }
        bool isLoad() const {
            return x(25, 3) == 0b100;
        }
        bool isStore() const {
            return x(25, 3) == 0b101;
        }
        // FIXME: This is only the matrix-matrix computation 
        bool isMatrixInstArith() const {
            return x(25, 3) == 0b000;
        }
        bool ismcfgki() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b0)&&(x(28, 3) == 0b000));
        }
        bool ismcfgmi() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b0)&&(x(28, 3) == 0b001));
        }
        bool ismcfgni() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b0)&&(x(28, 3) == 0b010));
        }
        bool ismcfgk() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b1)&&(x(28, 3) == 0b000));
        }
        bool ismcfgm() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b1)&&(x(28, 3) == 0b001));
        }
        bool ismcfgn() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b1)&&(x(28, 3) == 0b010));
        }
        bool ismcfg() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b1)&&(x(28, 3) == 0b111));
        }
        bool isSigned() const {
            return (x(7, 3) == 0b000);
        }
        // FIXME: This just indicate the element is 16bit, not float16
        bool isFP16() const {
            return x(10, 2) == 0b01;
        }
        //FIXME: This just indicate the element is 8bit, not int8
        bool isInt8() const {
            return x(10, 2) == 0b00;
        }

        /*Instruction encoding format*/
        /*FIXME: This part of function has been implemented in riscv/types.hh*/
        /*virtual uint32_t majorcode() const = 0;
        virtual uint32_t ms1() const*/

        // uint32_t bits() { 
        //     return b;
        // }
        uint64_t getPC()
        {
            return pc;
        }

        const char *mnemonic;
    private:
        uint64_t pc;
        const uint32_t b;
        uint32_t x(int lo, int len) const {
            return (b >> lo) & ((uint32_t(1) << len)-1);
        }
    };

    class RiscvMatrixInsn : public RiscvStaticInst
    {
    protected:
        // using RiscvStaticInst::RiscvStaticInst;
        RiscvMatrixInsn(const char *_mnemonic, ExtMachInst _machInst,OpClass __opClass) :
        RiscvStaticInst(_mnemonic, _machInst, __opClass),
        b(_machInst), mnemonic(_mnemonic){}
        ~RiscvMatrixInsn(){};

    public:
        RegIndex ms1() const { return x(18, 3);}
        RegIndex ms2() const { return x(21, 3);}
        RegIndex ms3() const {return x(7, 3);}
        RegIndex md() const { return x(15, 3);}
        RegIndex load_md() const { return x(7, 3);}
        RegIndex mrs1() const { return x(15, 5);}
        RegIndex mrs2() const { return x(20, 5);}
        RegIndex mrd() const { return x(7, 5);}
        uint8_t uimm7() const {return x(18, 7);}
        uint8_t getbits_0_7() const {return x(0, 8);}
        uint8_t getbits_8_15() const {return x(8, 8);}
        uint16_t getbits_16_31() const {return x(16, 16);}
        uint8_t getbits_31() const {return x(31, 1);}
        uint8_t getbits_18_24() const {return x(18, 7);}
    public:
        void advancePC(PCStateBase &pc) const override
        {
            pc.as<PCState>().advance();
        }
        std::string getName() const
        {
            return mnemonic;
        }
        virtual std::string generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const= 0;

        bool isMatrixInst() const {
            return x(0, 7) == 0b0101011;
           // return b.m_majorcode == 0b;
        }
        bool isMatrixConfig() const {
            return x(25, 3) == 0b111;
            //return b.muop == 0b111
        }
        // virtual bool isMatrixInstMem() const {
        //     return x(               )
        // }
        bool isLoad() const {
            return x(25, 3) == 0b100;
        }
        bool isStore() const {
            return x(25, 3) == 0b101;
        }
        // FIXME: This is only the matrix-matrix computation 
        bool isMatrixInstArith() const {
            return x(25, 3) == 0b000;
        }
        bool ismcfgki() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b0)&&(x(28, 3) == 0b000));
        }
        bool ismcfgmi() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b0)&&(x(28, 3) == 0b001));
        }
        bool ismcfgni() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b0)&&(x(28, 3) == 0b010));
        }
        bool ismcfgk() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b1)&&(x(28, 3) == 0b000));
        }
        bool ismcfgm() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b1)&&(x(28, 3) == 0b001));
        }
        bool ismcfgn() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b1)&&(x(28, 3) == 0b010));
        }
        bool ismcfg() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b1)&&(x(28, 3) == 0b111));
        }
        bool isSigned() const {
            return (x(7, 3) == 0b000);
        }
        // FIXME: This just indicate the element is 16bit, not float16
        bool isFP16() const {
            return x(10, 2) == 0b01;
        }
        //FIXME: This just indicate the element is 8bit, not int8
        bool isInt8() const {
            return x(10, 2) == 0b00;
        }

        /*Instruction encoding format*/
        /*FIXME: This part of function has been implemented in riscv/types.hh*/
        /*virtual uint32_t majorcode() const = 0;
        virtual uint32_t ms1() const*/

        // uint32_t bits() { 
        //     return b;
        // }
        uint64_t getPC()
        {
            return pc;
        }

        const char *mnemonic;
    private:
        uint64_t pc;
        const uint32_t b;
        uint32_t x(int lo, int len) const {
            return (b >> lo) & ((uint32_t(1) << len)-1);
        }
    };

} // namespace RiscvISA
} // namespace gem5

#endif