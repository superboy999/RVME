/*
 * @Author: superboy
 * @Date: 2024-09-08 22:08:46
 * @LastEditTime: 2024-09-20 18:55:17
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/exploration/256.c
 * 
 */

// 256*512*64
// 256*2048*512
// 1024*1024*256

#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"
#define N 256 //indicate the whole matrix length

static alignas(32) int8_t A1[16][32][N] = { [0 ... 16-1][0 ... 32-1][0 ... N-1] = 1 };
static alignas(32) int8_t B1[16][8][N] = { [0 ... 16-1][0 ... 8-1][0 ... N-1] = 1 };
static alignas(32) int8_t C11[32][8][64] = { [0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1 };
static alignas(32) int8_t C12[32][8][64] = { [0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1 };

static alignas(32) int8_t A2[64][16][N] = { [0 ... 64-1][0 ... 16-1][0 ... N-1] = 1 };
static alignas(32) int8_t B2[64][64][N] = { [0 ... 64-1][0 ... 64-1][0 ... N-1] = 1 };
static alignas(32) int8_t C21[16][64][64] = { [0 ... 16-1][0 ... 64-1][0 ... 64-1] = 1 };
static alignas(32) int8_t C22[16][64][64] = { [0 ... 16-1][0 ... 64-1][0 ... 64-1] = 1 };

static alignas(32) int8_t A3[32][64][N] = { [0 ... 32-1][0 ... 64-1][0 ... N-1] = 1 };
static alignas(32) int8_t B3[32][32][N] = { [0 ... 32-1][0 ... 32-1][0 ... N-1] = 1 };
static alignas(32) int8_t C31[64][32][64] = { [0 ... 64-1][0 ... 32-1][0 ... 64-1] = 1 };
static alignas(32) int8_t C32[64][32][64] = { [0 ... 64-1][0 ... 32-1][0 ... 64-1] = 1 };

static alignas(64) int8_t A4[96][32][N] = { [0 ... 96-1][0 ... 32-1][0 ... N-1] = 1 };
static alignas(64) int8_t B4[96][96][N] = { [0 ... 96-1][0 ... 96-1][0 ... N-1] = 1 };
static alignas(64) int8_t C41[32][96][64] = { [0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1 };
static alignas(64) int8_t C42[32][96][64] = { [0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1 };

static alignas(64) int8_t A5[24][32][N] = { [0 ... 24-1][0 ... 32-1][0 ... N-1] = 1 };
static alignas(64) int8_t B5[24][32][N] = { [0 ... 24-1][0 ... 32-1][0 ... N-1] = 1 };
static alignas(64) int8_t C51[32][32][64] = { [0 ... 32-1][0 ... 32-1][0 ... 64-1] = 1 };
static alignas(64) int8_t C52[32][32][64] = { [0 ... 32-1][0 ... 32-1][0 ... 64-1] = 1 };

static alignas(64) int8_t A6[2][1568][N] = { [0 ... 2-1][0 ... 1568-1][0 ... N-1] = 1 };
static alignas(64) int8_t B6[2][12][N] = { [0 ... 2-1][0 ... 12-1][0 ... N-1] = 1 };
static alignas(64) int8_t C61[1568][12][64] = { [0 ... 1568-1][0 ... 12-1][0 ... 64-1] = 1 };
static alignas(64) int8_t C62[1568][12][64] = { [0 ... 1568-1][0 ... 12-1][0 ... 64-1] = 1 };

static alignas(64) int8_t A7[2][392][N] = { [0 ... 2-1][0 ... 392-1][0 ... N-1] = 1 };
static alignas(64) int8_t B7[2][8][N] = { [0 ... 2-1][0 ... 8-1][0 ... N-1] = 1 };
static alignas(64) int8_t C71[392][8][64] = { [0 ... 392-1][0 ... 8-1][0 ... 64-1] = 1 };
static alignas(64) int8_t C72[392][8][64] = { [0 ... 392-1][0 ... 8-1][0 ... 64-1] = 1 };

static void whole_matmul(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+6*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+6*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+7*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+7*8*stride_post, stride_post);

    mstb(0, (uint64_t *)start_addr_m0+0*8*stride, stride);
}
static void whole_madd(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 8 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(8);
    mcfgni(8);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+6*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+6*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+7*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+7*8*stride_post, stride_post);

    mstb(0, (uint64_t *)start_addr_m0+0*8*stride, stride);
}
static void whole_matmul6(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(16);
    mcfgni(8);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*8*stride_post, stride_post);
}
static void whole_madd6(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 8 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(8);
    mcfgni(8);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*8*stride_post, stride_post);
}
int main()
{
    
    // for (int8_t k = 0; k < 16; k ++){
    //     for (int8_t i = 0; i < 32; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             whole_matmul(A1[k][i], B1[k][8*j], C11[i][8*j]);
    //         }
    //     }
    // }

    // for (int8_t k = 0; k < 15; k ++){
    //     for (int8_t i = 0; i < 32; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             whole_madd(C11[i][8*j], C12[i][8*j], C11[i][8*j]);
    //         }
    //     }
    // }
    //=============
    // for (int8_t k = 0; k < 64; k ++){
    //     for (int8_t i = 0; i < 16; i ++){
    //         for(int8_t j = 0; j < 8; j++){
    //             whole_matmul(A2[k][i], B2[k][8*j], C21[i][8*j]);
    //         }
    //     }
    // }

    // for (int8_t k = 0; k < 63; k ++){
    //     for (int8_t i = 0; i < 16; i ++){
    //         for(int8_t j = 0; j < 8; j++){
    //             whole_madd(C22[i][8*j], C21[i][8*j], C21[i][8*j]);
    //         }
    //     }
    // }
    //==============
    // for (int8_t k = 0; k < 32; k ++){
    //     for (int16_t i = 0; i < 64; i ++){
    //         for(int16_t j = 0; j < 4; j++){
    //             whole_matmul(A3[k][i], B3[k][8*j], C31[i][8*j]);
    //         }
    //     }
    // }

    // for (int8_t k = 0; k < 31; k ++){
    //     for (int16_t i = 0; i < 64; i ++){
    //         for(int16_t j = 0; j < 4; j++){
    //             whole_madd(C31[i][8*j], C32[i][8*j], C32[i][8*j]);
    //         }
    //     }
    // }
    // /===============
    // for (int16_t k = 0; k < 96; k ++){
    //     for (int16_t i = 0; i < 32; i ++){
    //         for(int16_t j = 0; j < 12; j++){
    //             whole_matmul(A4[k][i], B4[k][8*j], C41[i][8*j]);
    //         }
    //     }
    // }

    // for (int16_t k = 0; k < 95; k ++){
    //     for (int16_t i = 0; i < 32; i ++){
    //         for(int16_t j = 0; j < 12; j++){
    //             whole_madd(C41[i][8*j], C42[i][8*j], C42[i][8*j]);
    //         }
    //     }
    // }
    //===============
    // for (int8_t k = 0; k < 24; k ++){
    //     for (int8_t i = 0; i < 32; i ++){
    //         for(int8_t j = 0; j < 4; j++){
    //             whole_matmul(A5[k][i], B5[k][8*j], C52[i][8*j]);
    //         }
    //     }
    // }

    // for (int8_t k = 0; k < 23; k ++){
    //     for (int8_t i = 0; i < 32; i ++){
    //         for(int8_t j = 0; j < 4; j++){
    //             whole_madd(C52[i][8*j], C51[i][8*j], C52[i][8*j]);
    //         }
    //     }
    // }
    //==========================
        // for (int16_t i = 0; i < 1568; i ++){
        //     for(int16_t j = 0; j < 2; j++){
        //         whole_matmul6(A6[0][i], B6[0][8*j], C61[i][8*j]);
        //     }
        // }
    //========================
    for (int16_t k = 0; k < 2; k ++){
        for (int16_t i = 0; i < 392; i ++){
            for(int16_t j = 0; j < 1; j++){
                whole_matmul(A7[k][i], B7[k][8*j], C71[i][8*j]);
            }
        }
    }
        for (int16_t i = 0; i < 392; i ++){
            for(int16_t j = 0; j < 1; j++){
                whole_madd(C71[i][8*j], C72[i][8*j], C71[i][8*j]);
            }
        }

    return 0;
}cache_line_size