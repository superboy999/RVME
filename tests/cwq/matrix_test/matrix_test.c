#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#define N 8
static void mcfgk(uint16_t k_value)
{
    __asm__ __volatile__ ("mcfgk zero, %[src1]"
                            :
                            :[src1]"r"(k_value)
                        );
}

static void mcfgn(uint16_t n_value)
{
    __asm__ __volatile__ ("mcfgn zero, %[src1]"
                            :
                            :[src1]"r"(n_value)
                        );
}

static void mcfgm(uint16_t m_value)
{
    __asm__ __volatile__ ("mcfgm zero, %[src1]"
                            :
                            :[src1]"r"(m_value)
                        );
}

static void mcfgki(uint16_t k_value)
{
    __asm__ __volatile__ ("mcfgki zero, %0"
                            :
                            :"i"(k_value)
                        );
}

static void mcfgni(uint16_t n_value)
{
    __asm__ __volatile__ ("mcfgni zero, %0"
                            :
                            :"i"(n_value)
                        );
}

static void mcfgmi(uint16_t m_value)
{
    __asm__ __volatile__ ("mcfgmi zero, %0"
                            :
                            :"i"(m_value)
                        );
}

static void mldb(uint8_t mreg_idx, uint64_t *start_addr, uint64_t stride)
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

//Matrix multiplication: int8 datatype, unsigned x unsigned
static void matrixmul_int8_ss(int8_t destMregIdx, int8_t src1MregIdx, int8_t src2MregIdx)
{
    __asm__ __volatile__ ("mmaqa.b m%0, m%1, m%2"
                            : 
                            : "i" (destMregIdx), "i"(src2MregIdx), "i"(src1MregIdx));
}

// int main()
// {
//     printf("TEST matrix cfg ints start!\n");
//     uint16_t k = 4;
//     mcfgk(k);
//     uint16_t n = 5;
//     mcfgki(n);
//     printf("111\n");
//     printf("222\n");
//     return 0;
// }

int main()
{
    printf("TEST matrix mld ints start!\n");
    mcfgmi(2);
    mcfgki(4);
    mcfgni(2);
    // int8_t x[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // int8_t y[N] = {16, 12, 8, 4, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1};
    int8_t x[N] = {1, 2, 3, 4, 1, 2, 3, 4};
    int8_t y[N] = {1, 2, 3, 4, 1, 2, 3, 4};
    uint64_t stride = 4 * sizeof(int8_t);
    mldb(0, (uint64_t *)x, stride);
    mldb(1, (uint64_t *)y, stride);
    matrixmul_int8_ss(2, 0, 1);
    printf("TEST matrix mld ints finish!\n");
    return 0;
}