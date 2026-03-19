#ifndef __INST_H
#define __INST_H

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

// matrix config instructions
inline static void msettilek(uint16_t k_value)
{
    __asm__ __volatile__ ("msettilek %[src1]"
                            :
                            :[src1]"r"(k_value)
                        );
}

inline static void msettileki(uint16_t k_value)
{
    __asm__ __volatile__ ("msettileki %0"
                            :
                            :"i"(k_value)
                        );
}

inline static void msettilem(uint16_t m_value)
{
    __asm__ __volatile__ ("msettilem %[src1]"
                            :
                            :[src1]"r"(m_value)
                        );
}

inline static void msettilemi(uint16_t m_value)
{
    __asm__ __volatile__ ("msettilemi %0"
                            :
                            :"i"(m_value)
                        );
}

inline static void msettilen(uint16_t n_value)
{
    __asm__ __volatile__ ("msettilen %[src1]"
                            :
                            :[src1]"r"(n_value)
                        );
}

inline static void msettileni(uint16_t n_value)
{
    __asm__ __volatile__ ("msettileni %0"
                            :
                            :"i"(n_value)
                        );
}

// matrix multiply-accumulate instructions
inline static void mmacc_w_b(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mmacc.w.b m%0, m%1, m%2"
                            : 
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void mmaccu_w_b(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mmaccu.w.b m%0, m%1, m%2"
                            : 
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void mmaccus_w_b(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mmaccus.w.b m%0, m%1, m%2"
                            : 
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void mmaccsu_w_b(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mmaccsu.w.b m%0, m%1, m%2"
                            : 
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

// matrix load/store instructions
// rs1_value = memory address
// rs2_value = row byte stride
inline static void mlae8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mlae8 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value) : "memory"
                        );
}

inline static void mlbe8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mlbe8 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value) : "memory"
                        );
}

inline static void mlce32(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mlce32 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value) : "memory"
                        );
}

inline static void msae8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("msae8 m%1, (%2), %3"
                            : "=m" (*((uint8_t *)rs1_value))
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value) : "memory"
                        );
}

inline static void msbe8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("msbe8 m%1, (%2), %3"
                            : "=m" (*((uint8_t *)rs1_value))
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value) : "memory"
                        );
}

inline static void msce32(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("msce32 m%1, (%2), %3"
                            : "=m" (*((uint32_t *)rs1_value))
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value) : "memory"
                        );
}

inline static void mlate8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mlate8 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value) : "memory"
                        );
}

inline static void mlbte8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mlbte8 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value) : "memory"
                        );
}

inline static void mlcte32(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mlcte32 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value) : "memory"
                        );
}

inline static void msate8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("msate8 m%1, (%2), %3"
                            : "=m" (*((uint8_t *)rs1_value))
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value) : "memory"
                        );
}

inline static void msbte8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("msbte8 m%1, (%2), %3"
                            : "=m" (*((uint8_t *)rs1_value))
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value) : "memory"
                        );
}

inline static void mscte32(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mscte32 m%1, (%2), %3"
                            : "=m" (*((uint32_t *)rs1_value))
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value) : "memory"
                        );
}

// matrix move instructions
inline static void mmov_mm(uint8_t destMregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mmov.mm m%0, m%1"
                            : 
                            : "i" (destMregIdx), "i"(src1MregIdx)
                        );
}

inline static int32_t mmovb_x_m(uint8_t srcMregIdx, uint64_t rs1_value)
{
    int32_t rd_data;
    __asm__ __volatile__ ("mmovb.x.m %[d], m%[a], %[c]"
                            : [d] "=r" (rd_data)
                            : [a] "i" (srcMregIdx), [c] "r"(rs1_value)
                        );
    return rd_data;
}

inline static int32_t mmovw_x_m(uint8_t srcMregIdx, uint64_t rs1_value)
{
    int32_t rd_data;
    __asm__ __volatile__ ("mmovw.x.m %[d], m%[a], %[c]"
                            : [d] "=r" (rd_data)
                            : [a] "i" (srcMregIdx), [c] "r"(rs1_value)
                        );
    return rd_data;
}

inline static void mmovb_m_x(uint8_t destMregIdx, uint64_t rs2_value, uint64_t rs1_value)
{
    __asm__ __volatile__ ("mmovb.m.x m%0, %1, %2"
                            :
                            : "i" (destMregIdx), "r"(rs2_value), "r"(rs1_value)
                        );
}

inline static void mmovw_m_x(uint8_t destMregIdx, uint64_t rs2_value, uint64_t rs1_value)
{
    __asm__ __volatile__ ("mmovw.m.x m%0, %1, %2"
                            :
                            : "i" (destMregIdx), "r"(rs2_value), "r"(rs1_value)
                        );
}

// matrix arithmetic instructions
inline static void madd_w_mm(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("madd.w.mm m%0, m%1, m%2"
                            :
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void msub_w_mm(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("msub.w.mm m%0, m%1, m%2"
                            :
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void mmul_w_mm(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mmul.w.mm m%0, m%1, m%2"
                            :
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void mmulh_w_mm(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mmulh.w.mm m%0, m%1, m%2"
                            :
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void mmax_w_mm(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mmax.w.mm m%0, m%1, m%2"
                            :
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void mumax_w_mm(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mumax.w.mm m%0, m%1, m%2"
                            :
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void mmin_w_mm(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mmin.w.mm m%0, m%1, m%2"
                            :
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void mumin_w_mm(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mumin.w.mm m%0, m%1, m%2"
                            :
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void msrl_w_mm(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("msrl.w.mm m%0, m%1, m%2"
                            :
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void msll_w_mm(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("msll.w.mm m%0, m%1, m%2"
                            :
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void msra_w_mm(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("msra.w.mm m%0, m%1, m%2"
                            :
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}

inline static void madd_w_mv_i(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx, int8_t imm3)
{
    __asm__ __volatile__ ("madd.w.mv.i m%[a], m%[b], m%[c][%[d]]"
                            :
                            : [a]"i" (destMregIdx), [b]"i"(src2MregIdx), [c]"i"(src1MregIdx), [d]"i"(imm3)
                        );
}

inline static void msub_w_mv_i(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx, int8_t imm3)
{
    __asm__ __volatile__ ("msub.w.mv.i m%[a], m%[b], m%[c][%[d]]"
                            :
                            : [a]"i" (destMregIdx), [b]"i"(src2MregIdx), [c]"i"(src1MregIdx), [d]"i"(imm3)
                        );
}

inline static void mmul_w_mv_i(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx, int8_t imm3)
{
    __asm__ __volatile__ ("mmul.w.mv.i m%[a], m%[b], m%[c][%[d]]"
                            :
                            : [a]"i" (destMregIdx), [b]"i"(src2MregIdx), [c]"i"(src1MregIdx), [d]"i"(imm3)
                        );
}

inline static void mmulh_w_mv_i(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx, int8_t imm3)
{
    __asm__ __volatile__ ("mmulh.w.mv.i m%[a], m%[b], m%[c][%[d]]"
                            :
                            : [a]"i" (destMregIdx), [b]"i"(src2MregIdx), [c]"i"(src1MregIdx), [d]"i"(imm3)
                        );
}

inline static void mmax_w_mv_i(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx, int8_t imm3)
{
    __asm__ __volatile__ ("mmax.w.mv.i m%[a], m%[b], m%[c][%[d]]"
                            :
                            : [a]"i" (destMregIdx), [b]"i"(src2MregIdx), [c]"i"(src1MregIdx), [d]"i"(imm3)
                        );
}

inline static void mumax_w_mv_i(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx, int8_t imm3)
{
    __asm__ __volatile__ ("mumax.w.mv.i m%[a], m%[b], m%[c][%[d]]"
                            :
                            : [a]"i" (destMregIdx), [b]"i"(src2MregIdx), [c]"i"(src1MregIdx), [d]"i"(imm3)
                        );
}

inline static void mmin_w_mv_i(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx, int8_t imm3)
{
    __asm__ __volatile__ ("mmin.w.mv.i m%[a], m%[b], m%[c][%[d]]"
                            :
                            : [a]"i" (destMregIdx), [b]"i"(src2MregIdx), [c]"i"(src1MregIdx), [d]"i"(imm3)
                        );
}

inline static void mumin_w_mv_i(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx, int8_t imm3)
{
    __asm__ __volatile__ ("mumin.w.mv.i m%[a], m%[b], m%[c][%[d]]"
                            :
                            : [a]"i" (destMregIdx), [b]"i"(src2MregIdx), [c]"i"(src1MregIdx), [d]"i"(imm3)
                        );
}

inline static void msrl_w_mv_i(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx, int8_t imm3)
{
    __asm__ __volatile__ ("msrl.w.mv.i m%[a], m%[b], m%[c][%[d]]"
                            :
                            : [a]"i" (destMregIdx), [b]"i"(src2MregIdx), [c]"i"(src1MregIdx), [d]"i"(imm3)
                        );
}

inline static void msll_w_mv_i(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx, int8_t imm3)
{
    __asm__ __volatile__ ("msll.w.mv.i m%[a], m%[b], m%[c][%[d]]"
                            :
                            : [a]"i" (destMregIdx), [b]"i"(src2MregIdx), [c]"i"(src1MregIdx), [d]"i"(imm3)
                        );
}

inline static void msra_w_mv_i(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx, int8_t imm3)
{
    __asm__ __volatile__ ("msra.w.mv.i m%[a], m%[b], m%[c][%[d]]"
                            :
                            : [a]"i" (destMregIdx), [b]"i"(src2MregIdx), [c]"i"(src1MregIdx), [d]"i"(imm3)
                        );
}

// SPM instructions
// rs1_value = spm address
// rs2_value = row byte stride
inline static void mlae8_spm(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value, uint8_t mode)
{
    assert(destMregIdx < 4);
    __asm__ __volatile__ ("mlae8.spm m%0, (%1), %2, %3"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value), "i"(mode) : "memory"
                        );
}

inline static void mlbe8_spm(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value, uint8_t mode)
{
    assert(destMregIdx < 4);
    __asm__ __volatile__ ("mlbe8.spm m%0, (%1), %2, %3"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value), "i"(mode) : "memory"
                        );
}

inline static void mlce32_spm(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value, uint8_t mode)
{
    assert(destMregIdx >= 4 && destMregIdx < 8);
    __asm__ __volatile__ ("mlce32.spm m%0, (%1), %2, %3"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value), "i"(mode) : "memory"
                        );
}

inline static void msae8_spm(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value, uint8_t mode)
{
    __asm__ __volatile__ ("msae8.spm m%1, (%2), %3, %4"
                            : "=m" (*((uint8_t *)rs1_value))
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value), "i"(mode) : "memory"
                        );
}

inline static void msbe8_spm(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value, uint8_t mode)
{
    __asm__ __volatile__ ("msbe8.spm m%1, (%2), %3, %4"
                            : "=m" (*((uint8_t *)rs1_value))
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value), "i"(mode) : "memory"
                        );
}

inline static void msce32_spm(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value, uint8_t mode)
{
    __asm__ __volatile__ ("msce32.spm m%1, (%2), %3, %4"
                            : "=m" (*((uint32_t *)rs1_value))
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value), "i"(mode) : "memory"
                        );
}

inline static void mlate8_spm(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value, uint8_t mode)
{
    __asm__ __volatile__ ("mlate8.spm m%0, (%1), %2, %3"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value), "i"(mode) : "memory"
                        );
}

inline static void mlbte8_spm(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value, uint8_t mode)
{
    __asm__ __volatile__ ("mlbte8.spm m%0, (%1), %2, %3"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value), "i"(mode) : "memory"
                        );
}

inline static void mlcte32_spm(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value, uint8_t mode)
{
    __asm__ __volatile__ ("mlcte32.spm m%0, (%1), %2, %3"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value), "i"(mode) : "memory"
                        );
}

inline static void msate8_spm(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value, uint8_t mode)
{
    __asm__ __volatile__ ("msate8.spm m%1, (%2), %3, %4"
                            : "=m" (*((uint8_t *)rs1_value))
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value), "i"(mode) : "memory"
                        );
}

inline static void msbte8_spm(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value, uint8_t mode)
{
    __asm__ __volatile__ ("msbte8.spm m%1, (%2), %3, %4"
                            : "=m" (*((uint8_t *)rs1_value))
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value), "i"(mode) : "memory"
                        );
}

inline static void mscte32_spm(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value, uint8_t mode)
{
    __asm__ __volatile__ ("mscte32.spm m%1, (%2), %3, %4"
                            : "=m" (*((uint32_t *)rs1_value))
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value), "i"(mode) : "memory"
                        );
}

// rs1_value = memory address
// rs2_value = spm address
// uimm5 = size (in bytes)/ cache line size
inline static void dmaload_spm(uint64_t rs1_value, uint64_t rs2_value, uint8_t uimm5)
{
    __asm__ __volatile__ ("load.spm %0, %1, %2"
                            : 
                            : "r"(rs1_value), "r"(rs2_value), "i"(uimm5) : "memory"
                        );
}

inline static void dmastore_spm(uint64_t rs1_value, uint64_t rs2_value, uint8_t uimm5)
{
    __asm__ __volatile__ ("store.spm %1, %2, %3"
                            : "=m" (*((void *)rs1_value))
                            : "r"(rs1_value), "r"(rs2_value), "i"(uimm5) : "memory"
                        );
}

// float
inline static void mfmacc_s_e5(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mfmacc.s.e5 m%0, m%1, m%2"
                            : 
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}
inline static void mfmacc_s_e4(uint8_t destMregIdx, uint8_t src2MregIdx, uint8_t src1MregIdx)
{
    __asm__ __volatile__ ("mfmacc.s.e4 m%0, m%1, m%2"
                            : 
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx)
                        );
}
inline static void mflae8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mflae8 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value)
                        );
}
inline static void mflbe8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mflbe8 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value)
                        );
}
inline static void mflce32(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mflce32 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value)
                        );
}
inline static void mfsae8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mfsae8 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value)
                        );
}
inline static void mfsbe8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{           
    __asm__ __volatile__ ("mfsbe8 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value)
                        );
}
inline static void mfsce32(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mfsce32 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value)
                        );
}
inline static void mflate8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mflate8 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value)
                        );
}
inline static void mflbte8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mflbte8 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value)
                        );
}
inline static void mflcte32(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mflcte32 m%0, (%1), %2"
                            : 
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value)
                        );
}
inline static void mfsate8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mfsate8 m%0, (%1), %2"
                            :
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value)
                        );
}
inline static void mfsbte8(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mfsbte8 m%0, (%1), %2"
                            :
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value)
                        );
}
inline static void mfscte32(uint8_t destMregIdx, uint64_t rs1_value, uint64_t rs2_value)
{
    __asm__ __volatile__ ("mfscte32 m%0, (%1), %2"
                            :
                            : "i" (destMregIdx), "r"(rs1_value), "r"(rs2_value)
                        );
}

//  new instructions
inline static void mredcmax_w(uint8_t destMregIdx, uint8_t srcMregIdx)
{
    __asm__ __volatile__ ("mredcmax.w m%0, m%1"
                            : 
                            : "i" (destMregIdx), "i"(srcMregIdx)
                        );
}
inline static void mredcadd_w_i(uint8_t destMregIdx, uint8_t srcMregIdx, uint8_t uimm3)
{
    __asm__ __volatile__ ("mredcadd.w.i m%[a][%[c]], m%[b]"
                            :
                            : [a] "i" (destMregIdx), [b] "i"(srcMregIdx), [c] "i"(uimm3)
                        );
}
inline static void mredcmax_w_i(uint8_t destMregIdx, uint8_t srcMregIdx, uint8_t uimm3)
{
    __asm__ __volatile__ ("mredcmax.w.i m%[a][%[c]], m%[b]"
                            :
                            : [a] "i" (destMregIdx), [b] "i"(srcMregIdx), [c] "i"(uimm3)
                        );
}
inline static void mredcadd_w(uint8_t destMregIdx, uint8_t srcMregIdx)
{
    __asm__ __volatile__ ("mredcadd.w m%0, m%1"
                            :
                            : "i" (destMregIdx), "i"(srcMregIdx)
                        );
}
inline static void mzero(uint8_t destMregIdx)
{
    __asm__ __volatile__ ("mzero m%0"
                            :
                            : "i"(destMregIdx)
                        );
}
inline static void minv_w_i(uint8_t destMregIdx, uint8_t srcMregIdx, int8_t uimm3)
{
    __asm__ __volatile__ ("minv.w.i m%[a], m%[b][%[c]]"
                            :
                            : [a] "i" (destMregIdx), [b] "i"(srcMregIdx), [c] "i"(uimm3)
                        );
}
inline static void msync_spm()
{
    __asm__ __volatile__ ("msync.spm"
                            :
                            :
                        );
}
inline static void mlut_w_i(uint8_t destMregIdx, uint8_t srcMregIdx, int8_t uimm3, int8_t type)
{
    __asm__ __volatile__ ("mlut.w.i m%[a], m%[b][%[c]], %[d]"
                            :
                            : [a] "i" (destMregIdx), [b] "i"(srcMregIdx), [c] "i"(uimm3), [d] "i"(type)
                        );
}

#endif