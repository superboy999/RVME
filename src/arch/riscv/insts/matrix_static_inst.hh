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
        RiscvMatrixInst(const char *_mnemonic, ExtMachInst _machInst, OpClass __opClass) :
        RiscvStaticInst(_mnemonic, _machInst, __opClass),
        mnemonic(_mnemonic), b(_machInst){}
        ~RiscvMatrixInst(){}

    public:
        RegIndex ms1() const { return x(15, 3);}
        RegIndex ms2() const { return x(20, 3);}
        RegIndex ms3() const {return x(7, 3);}
        RegIndex md() const { return x(7, 3);}
        RegIndex mrs1() const { return x(15, 5);}
        RegIndex mrs2() const { return x(20, 5);}
        RegIndex mrd() const { return x(7, 5);}
        uint8_t uimm3() const {return x(23, 3);}
        uint32_t uimm10() const {return x(15, 10);}
        uint8_t mfunct4() const {return x(28, 4);}
        uint8_t bit25_23() const {return x(23, 3);}
        uint8_t d_size() const {return x(10, 2);}
        uint8_t s_size() const {return x(18, 2);}
        uint8_t uimm5() const {return x(7, 5);}
        uint8_t getbits_8_15() const {return x(8, 8);}
        uint8_t getbits_0_7() const {return x(0, 8);}
        uint16_t getbits_16_31() const {return x(16, 8);}
        uint16_t getbits_18_24() const {return x(18, 7);}
        
        uint8_t getbits_23_25() const {return x(23, 3);}
        uint16_t getbits_25() const {return x(25, 1);}
        uint8_t spm_mode() const {return x(10, 2);}
        uint8_t bit29_28() const {return x(28, 2);}

        
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

        bool isMatrix() const {
            return x(0, 7) == 0b0101011;
        }
        bool isMatrixConfig() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b00));
        }
        bool isSigned() const {
            return ismmacc_w_b() || ismredcadd() || ismlut() || ismredcmax() || (isMatrixElementWise() && !ismumax() && !ismumin());
        }
        bool isMatrixInstArith() const {
            return (((x(12, 3) == 0b000) && (x(26, 2) == 0b10)) ||
                   ((x(12, 3) == 0b001) && (x(26, 2) == 0b01)));
        }
        bool isMatrixElementWise() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01));
        }
        bool isMatrixDataMove() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b11));
        }
        bool isMatrixBroadcast() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b11) && (x(28, 4) == 0b1000) && (x(25, 1) == 0b1) && (x(23, 2) == 0b10));
        }
        bool isMatrixRedc() const {
            return ((x(12, 3) == 0b101) && (x(26, 2) == 0b01));
        }
        bool isLoad() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b0));
        }
        bool isStore() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b1));
        }
        bool isMISC() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b11));
        }
        // === these has been allocated into Config instructions
        bool ismcfgki() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b00) && (x(28, 4) == 0b0001) && (x(25, 1) == 0b0));
        }
        bool ismcfgk() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b00) && (x(28, 4) == 0b0001) && (x(25, 1) == 0b1));
        }
        bool ismcfgmi() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b00) && (x(28, 4) == 0b0010) && (x(25, 1) == 0b0));
        }
        bool ismcfgm() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b00) && (x(28, 4) == 0b0010) && (x(25, 1) == 0b1));
        }
        bool ismcfgni() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b00) && (x(28, 4) == 0b0011) && (x(25, 1) == 0b0));
        }
        bool ismcfgn() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b00) && (x(28, 4) == 0b0011) && (x(25, 1) == 0b1));
        }

        bool ismcfg() const {
            return ((x(25, 3) == 0b111)&&(x(31, 1) == 0b1)&&(x(28, 3) == 0b111));
        }
        // === these has been allocated into Arithmetic instructions
        bool ismmacc() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b10));
        }
        bool ismmacc_w_b() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b10) && (x(28, 4) == 0b0001) && (x(23, 3) == 0b011)
                    && (x(18, 2) == 0b00) && (x(10, 2) == 0b10));
        }
        bool ismmaccu_w_b() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b10) && (x(28, 4) == 0b0001) && (x(23, 3) == 0b000)
                    && (x(18, 2) == 0b00) && (x(10, 2) == 0b10));
        }
        bool ismmaccus_w_b() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b10) && (x(28, 4) == 0b0001) && (x(23, 3) == 0b001)
                    && (x(18, 2) == 0b00) && (x(10, 2) == 0b10));
        }
        bool ismmaccsu_w_b() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b10) && (x(28, 4) == 0b0001) && (x(23, 3) == 0b010)
                    && (x(18, 2) == 0b00) && (x(10, 2) == 0b10));
        }
        bool ismadd() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0000));
        }
        bool ismadd_w_mm() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0000)
                    && (x(23, 3) == 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismadd_w_mv_i() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0000)
                    && (x(23, 3) != 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismsub() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0001));
        }
        bool ismsub_w_mm() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0001)
                    && (x(23, 3) == 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismsub_w_mv_i() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0001)
                    && (x(23, 3) != 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismmul() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0010));
        }
        bool ismmul_w_mm() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0010)
                    && (x(23, 3) == 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismmul_w_mv_i() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0010)
                    && (x(23, 3) != 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismmulh() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0011));
        }
        bool ismmulh_w_mm() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0011)
                    && (x(23, 3) == 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismmulh_w_mv_i() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0011)
                    && (x(23, 3) != 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismmax() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0100));
        }
        bool ismmax_w_mm() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0100)
                    && (x(23, 3) == 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismmax_w_mv_i() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0100)
                    && (x(23, 3) != 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismumax() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0101));
        }
        bool ismumax_w_mm() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0101)
                    && (x(23, 3) == 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismumax_w_mv_i() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0101)
                    && (x(23, 3) != 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismmin() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0110));
        }
        bool ismmin_w_mm() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0110)
                    && (x(23, 3) == 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismmin_w_mv_i() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0110)
                    && (x(23, 3) != 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismumin() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0111));
        }
        bool ismumin_w_mm() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0111)
                    && (x(23, 3) == 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismumin_w_mv_i() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0111)
                    && (x(23, 3) != 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismsrl() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b1000));
        }
        bool ismsrl_w_mm() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b1000)
                    && (x(23, 3) == 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismsrl_w_mv_i() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b1000)
                    && (x(23, 3) != 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismsll() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b1001));
        }
        bool ismsll_w_mm() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b1001)
                    && (x(23, 3) == 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismsll_w_mv_i() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b1001)
                    && (x(23, 3) != 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismsra() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b1010));
        }
        bool ismsra_w_mm() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b1010)
                    && (x(23, 3) == 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        bool ismsra_w_mv_i() const {
            return ((x(12, 3) == 0b001) && (x(26, 2) == 0b01) && (x(28, 4) == 0b1010)
                    && (x(23, 3) != 0b111) && (x(18, 2) == 0b10) && (x(10, 2) == 0b10));
        }
        // === these has been allocated into Load\Store instructions
        bool ismload() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b0));
        }
        bool ismlae8() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b0)
                    && (x(28, 4) == 0b0000) && (x(10, 2) == 0b00));
        }
        bool ismlbe8() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b0)
                    && (x(28, 4) == 0b0001) && (x(10, 2) == 0b00));
        }
        bool ismlce32() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b0)
                    && (x(28, 4) == 0b0010) && (x(10, 2) == 0b10));
        }
        bool ismlate8() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b0)
                    && (x(28, 4) == 0b0100) && (x(10, 2) == 0b00));
        }
        bool ismlbte8() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b0)
                    && (x(28, 4) == 0b0101) && (x(10, 2) == 0b00));
        }
        bool ismlcte32() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b0)
                    && (x(28, 4) == 0b0110) && (x(10, 2) == 0b10));
        }
        bool ismstore() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b1));
        }
        bool ismsae8() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b1)
                    && (x(28, 4) == 0b0000) && (x(10, 2) == 0b00));
        }
        bool ismsbe8() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b1)
                    && (x(28, 4) == 0b0001) && (x(10, 2) == 0b00));
        }
        bool ismsce32() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b1)
                    && (x(28, 4) == 0b0010) && (x(10, 2) == 0b10));
        }
        bool ismsate8() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b1)
                    && (x(28, 4) == 0b0100) && (x(10, 2) == 0b00));
        }
        bool ismsbte8() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b1)
                    && (x(28, 4) == 0b0101) && (x(10, 2) == 0b00));
        }
        bool ismscte32() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b01) && (x(25, 1) == 0b1)
                    && (x(28, 4) == 0b0110) && (x(10, 2) == 0b10));
        }
        bool ismloadspm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b0));
        }
        bool ismlae8_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b0)
                    && (x(28, 4) == 0b0000));
        }
        bool ismlbe8_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b0)
                    && (x(28, 4) == 0b0001));
        }
        bool ismlce32_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b0)
                    && (x(28, 4) == 0b0010));
        }
        bool ismlate8_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b0)
                    && (x(28, 4) == 0b0100));
        }
        bool ismlbte8_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b0)
                    && (x(28, 4) == 0b0101));
        }
        bool ismlcte32_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b0)
                    && (x(28, 4) == 0b0110));
        }
        bool ismstorespm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b1));
        }
        bool ismsae8_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b1)
                    && (x(28, 4) == 0b0000));
        }
        bool ismsbe8_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b1)
                    && (x(28, 4) == 0b0001));
        }
        bool ismsce32_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b1)
                    && (x(28, 4) == 0b0010));
        }
        bool ismsate8_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b1)
                    && (x(28, 4) == 0b0100));
        }
        bool ismsbte8_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b1)
                    && (x(28, 4) == 0b0101));
        }
        bool ismscte32_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b00) && (x(25, 1) == 0b1)
                    && (x(28, 4) == 0b0110));
        }
        bool isspmdmaload() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b01) && (x(25, 1) == 0b0));
        }
        bool isspmdmastore() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b01) && (x(25, 1) == 0b1));
        }
        // === these has been allocated into MISC instructions
        bool ismmov_mm() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b11) && (x(28, 4) == 0b0001) && (x(25, 1) == 0b0));
        }
        bool ismmovb_x_m() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b11) && (x(28, 4) == 0b0010) && (x(25, 1) == 0b0)
                && (x(23, 2) == 0b00));
        }
        bool ismmovw_x_m() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b11) && (x(28, 4) == 0b0010) && (x(25, 1) == 0b0)
                && (x(23, 2) == 0b10));
        }
        bool ismmovb_m_x() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b11) && (x(28, 4) == 0b0011) && (x(25, 1) == 0b1)
                && (x(10, 2) == 0b00));
        }
        bool ismmovw_m_x() const {
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b11) && (x(28, 4) == 0b0011) && (x(25, 1) == 0b1)
                && (x(10, 2) == 0b10));
        }
        // new instructions
        bool ismredcadd() const {
            return ((x(12, 3) == 0b101) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0000));
        }
        bool ismredcmax() const {
            return ((x(12, 3) == 0b101) && (x(26, 2) == 0b01) && (x(28, 4) == 0b0100));
        }
        bool ismlut() const {
            return ((x(12, 3) == 0b101) && (x(26, 2) == 0b00) && (x(28, 4) == 0b0000));
        }
        bool ismsync_spm() const {
            return ((x(12, 3) == 0b100) && (x(26, 2) == 0b11));
        }

        //FIXME: mzero encoding fix
        bool ismzero() const{
            return ((x(12, 3) == 0b000) && (x(26, 2) == 0b11) && (x(28, 4) == 0b0100) && (x(25, 1) == 0b1)
                && (x(23, 2) == 0b10));
        }
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

#endif // __ARCH_RISCV_MATRIX_STATIC_INST_HH__
