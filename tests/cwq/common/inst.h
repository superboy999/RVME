/*** 
 * @Author: superboy
 * @Date: 2024-06-19 15:27:40
 * @LastEditTime: 2025-04-09 03:09:09
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/common/inst.h
 * @
 */

inline static void mcfgk(uint16_t k_value)
{
    __asm__ __volatile__ ("mcfgk zero, %[src1]"
                            :
                            :[src1]"r"(k_value)
                        );
}

inline static void mcfgn(uint16_t n_value)
{
    __asm__ __volatile__ ("mcfgn zero, %[src1]"
                            :
                            :[src1]"r"(n_value)
                        );
}

inline static void mcfgm(uint16_t m_value)
{
    __asm__ __volatile__ ("mcfgm zero, %[src1]"
                            :
                            :[src1]"r"(m_value)
                        );
}

inline static void mcfgki(uint16_t k_value)
{
    __asm__ __volatile__ ("mcfgki zero, %0"
                            :
                            :"i"(k_value)
                        );
}

inline static void mcfgni(uint16_t n_value)
{
    __asm__ __volatile__ ("mcfgni zero, %0"
                            :
                            :"i"(n_value)
                        );
}

inline static void mcfgmi(uint16_t m_value)
{
    __asm__ __volatile__ ("mcfgmi zero, %0"
                            :
                            :"i"(m_value)
                        );
}

inline static void mldb(uint8_t mreg_idx, uint64_t *start_addr, uint64_t stride)
{
    if (mreg_idx == 0) {
        __asm__ __volatile__ ("mldb m0, %0, (%1)"
                                :
                                : "r"(stride), "r"(start_addr));
    } else if (mreg_idx == 1) {
        __asm__ __volatile__ ("mldb m1, %0, (%1)"
                                :
                                : "r"(stride), "r"(start_addr));
    } else if (mreg_idx == 2) {
        __asm__ __volatile__ ("mldb m2, %0, (%1)"
                                :
                                : "r"(stride), "r"(start_addr));
    } else if (mreg_idx == 3) {
        __asm__ __volatile__ ("mldb m3, %0, (%1)"
                                :
                                : "r"(stride), "r"(start_addr));
    } else if (mreg_idx == 4) {
        __asm__ __volatile__ ("mldb m4, %0, (%1)"
                                :
                                : "r"(stride), "r"(start_addr));
    } else if (mreg_idx == 5) {
        __asm__ __volatile__ ("mldb m5, %0, (%1)"
                                :
                                : "r"(stride), "r"(start_addr));
    } else if (mreg_idx == 6) {
        __asm__ __volatile__ ("mldb m6, %0, (%1)"
                                :
                                : "r"(stride), "r"(start_addr));
    } else if (mreg_idx == 7) {
        __asm__ __volatile__ ("mldb m7, %0, (%1)"
                                :
                                : "r"(stride), "r"(start_addr));
    }
}

inline static void mldb_m0(uint64_t *start_addr, uint64_t stride)
{
    __asm__ __volatile__ ("mldb m0, %0, (%1)"
                            :
                            : "r"(stride), "r"(start_addr));
}

inline static void mldb_m1(uint64_t *start_addr, uint64_t stride)
{
    __asm__ __volatile__ ("mldb m1, %0, (%1)"
                                :
                            : "r"(stride), "r"(start_addr));
}

inline static void mldb_m2(uint64_t *start_addr, uint64_t stride)
{
    __asm__ __volatile__ ("mldb m2, %0, (%1)"
                                :
                            : "r"(stride), "r"(start_addr));
}

inline static void mldb_m3(uint64_t *start_addr, uint64_t stride)
{
    __asm__ __volatile__ ("mldb m3, %0, (%1)"
                                :
                            : "r"(stride), "r"(start_addr));
}

inline static void mldb_m4(uint64_t *start_addr, uint64_t stride)
{
    __asm__ __volatile__ ("mldb m4, %0, (%1)"
                                :
                            : "r"(stride), "r"(start_addr));
}

inline static void mldb_m5(uint64_t *start_addr, uint64_t stride)
{
    __asm__ __volatile__ ("mldb m5, %0, (%1)"
                                :
                            : "r"(stride), "r"(start_addr));
}

inline static void mldb_m6(uint64_t *start_addr, uint64_t stride)
{
    __asm__ __volatile__ ("mldb m6, %0, (%1)"
                                :
                            : "r"(stride), "r"(start_addr));
}

inline static void mldb_m7(uint64_t *start_addr, uint64_t stride)
{
    __asm__ __volatile__ ("mldb m7, %0, (%1)"
                                :
                            : "r"(stride), "r"(start_addr));
}

//Matrix multiplication: int8 datatype, signed x signed
inline static void matrixmul_int8_ss(int8_t destMregIdx, int8_t src1MregIdx, int8_t src2MregIdx)
{
    __asm__ __volatile__ ("mmaqa.b m%0, m%1, m%2"
                            : 
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx));
}

inline static void mstb(int8_t srcMregIdx, uint64_t *start_addr, uint64_t stride)
{
    __asm__ __volatile__ ("mstb m%0, %1, (%2)"
                                :
                            : "i" (srcMregIdx), "r"(stride), "r"(start_addr));
}

inline static void madd(int8_t destMregIdx, int8_t src1MregIdx, int8_t src2MregIdx)
{
    __asm__ __volatile__ ("madd.s.mm m%0, m%1, m%2"
                            : 
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx));
}

static void inner_mmul(uint64_t *start_addr_A, uint64_t *start_addr_B)
{
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    // mcfgmi(8);
    // mcfgki(32);
    // mcfgni(8);
    mldb_m0((uint64_t *)start_addr_A, stride);
    mldb_m2((uint64_t *)start_addr_B, stride);
    matrixmul_int8_ss(4, 0, 2);
    mldb_m1((uint64_t *)start_addr_A+1*stride_post*stride, stride);
    matrixmul_int8_ss(5, 1, 2);
    mldb_m3((uint64_t *)start_addr_B+1*stride_post*stride, stride);
    matrixmul_int8_ss(6, 1, 3);
    matrixmul_int8_ss(7, 0, 3);
}

static void outer_firstN_init(uint64_t *start_addr_A, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size 
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    uint64_t stride_A = stride * stride_post;
    uint64_t stride_C = stride * stride_post;
    // mcfgmi(8);
    // mcfgki(32);
    // mcfgni(8);
    mldb_m0((uint64_t *)start_addr_A, stride);
    mldb_m3((uint64_t *)start_addr_B, stride);
    matrixmul_int8_ss(5, 0, 3);
    mstb(5,(uint64_t *)start_addr_C+0*stride_C, stride_post);
    mldb_m1((uint64_t *)start_addr_A+0*stride_A, stride);
    matrixmul_int8_ss(6, 1, 3);
    mstb(6,(uint64_t *)start_addr_C+0*stride_C, stride_post);
    mldb_m2((uint64_t *)start_addr_A+0*stride_A, stride);
    matrixmul_int8_ss(7, 2, 3);
    mstb(7,(uint64_t *)start_addr_C+0*stride_C, stride_post);
}

static void outer_firstN(uint64_t *start_addr_A, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size 
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    uint64_t stride_A = stride * stride_post;
    uint64_t stride_C = stride * stride_post;
    mldb_m3((uint64_t *)start_addr_B, stride);
    matrixmul_int8_ss(5, 0, 3);
    mstb(5,(uint64_t *)start_addr_C+0*stride_C, stride_post);
    matrixmul_int8_ss(6, 1, 3);
    mstb(6,(uint64_t *)start_addr_C+0*stride_C, stride_post);
    matrixmul_int8_ss(7, 2, 3);
    mstb(7,(uint64_t *)start_addr_C+0*stride_C, stride_post);
}

static void outer_firstM_init(uint64_t *start_addr_A, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size 
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    uint64_t stride_B = stride * stride_post;
    uint64_t stride_C = stride * stride_post;
    // mcfgmi(8);
    // mcfgki(32);
    // mcfgni(8);
    mldb_m0((uint64_t *)start_addr_B, stride);
    mldb_m4((uint64_t *)start_addr_A, stride);
    matrixmul_int8_ss(5, 0, 4);
    mstb(5,(uint64_t *)start_addr_C+0*stride_C, stride);
    mldb_m1((uint64_t *)start_addr_B+1*stride_B, stride);
    matrixmul_int8_ss(6, 1, 4);
    mstb(6,(uint64_t *)start_addr_C+1*stride_C, stride);
    mldb_m2((uint64_t *)start_addr_B+2*stride_B, stride);
    matrixmul_int8_ss(7, 2, 4);
    mstb(7,(uint64_t *)start_addr_C+2*stride_C, stride);
}

static void outer_firstM(uint64_t *start_addr_A, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size 
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    uint64_t stride_B = stride * stride_post;
    uint64_t stride_C = stride * stride_post;

    mldb_m4((uint64_t *)start_addr_A, stride);
    matrixmul_int8_ss(5, 0, 4);
    mstb(5,(uint64_t *)start_addr_C+0*stride_C, stride);
    matrixmul_int8_ss(6, 1, 4);
    mstb(6,(uint64_t *)start_addr_C+1*stride_C, stride);
    matrixmul_int8_ss(7, 2, 4);
    mstb(7,(uint64_t *)start_addr_C+2*stride_C, stride);
}


static void outer_mmul2(uint64_t *start_addr_A, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size 
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);
    mldb_m0((uint64_t *)start_addr_A, stride);
    mldb_m4((uint64_t *)start_addr_B, stride);
    matrixmul_int8_ss(5, 0, 4);
    mstb(4,(uint64_t *)start_addr_C+0*stride_post*stride_post, stride_post);
    mldb_m1((uint64_t *)start_addr_A+1*stride_post*stride, stride);
    matrixmul_int8_ss(6, 1, 4);
    mstb(4,(uint64_t *)start_addr_C+1*stride_post*stride_post, stride_post);
}

static void outer_mmul1(uint64_t *start_addr_A, uint64_t *start_addr_B, uint64_t *start_addr_C, uint64_t C_stride)
{
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size 
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);
    mldb_m0((uint64_t *)start_addr_A, stride);
    mldb_m4((uint64_t *)start_addr_B, stride);
    matrixmul_int8_ss(5, 0, 4);
    mstb(4,(uint64_t *)start_addr_C+0*C_stride, stride);
}