/*
 * @Author: superboy
 * @Date: 2024-09-09 02:13:57
 * @LastEditTime: 2024-09-18 19:05:45
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/exploration/512.c
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
#define N 1024 //indicate the whole matrix length

static alignas(64) int8_t A1[8][16][N] = { [0 ... 8-1][0 ... 16-1][0 ... N-1] = 1 };
static alignas(64) int8_t B1[8][4][N] = { [0 ... 8-1][0 ... 4-1][0 ... N-1] = 1 };
static alignas(64) int8_t C11[16][4][256] = { [0 ... 16-1][0 ... 4-1][0 ... 256-1] = 1 };
static alignas(64) int8_t C12[16][4][256] = { [0 ... 16-1][0 ... 4-1][0 ... 256-1] = 1 };

static alignas(64) int8_t A2[32][8][N] = { [0 ... 32-1][0 ... 8-1][0 ... N-1] = 1 };
static alignas(64) int8_t B2[32][32][N] = { [0 ... 32-1][0 ... 32-1][0 ... N-1] = 1 };
static alignas(64) int8_t C21[8][32][256] = { [0 ... 8-1][0 ... 32-1][0 ... 256-1] = 1 };
static alignas(64) int8_t C22[8][32][256] = { [0 ... 8-1][0 ... 32-1][0 ... 256-1] = 1 };

static alignas(64) int8_t A3[16][32][N] = { [0 ... 16-1][0 ... 32-1][0 ... N-1] = 1 };
static alignas(64) int8_t B3[16][16][N] = { [0 ... 16-1][0 ... 16-1][0 ... N-1] = 1 };
static alignas(64) int8_t C31[32][16][256] = { [0 ... 32-1][0 ... 16-1][0 ... 256-1] = 1 };
static alignas(64) int8_t C32[32][16][256] = { [0 ... 32-1][0 ... 16-1][0 ... 256-1] = 1 };

static alignas(64) int8_t A4[48][16][N] = { [0 ... 48-1][0 ... 16-1][0 ... N-1] = 1 };
static alignas(64) int8_t B4[48][48][N] = { [0 ... 48-1][0 ... 48-1][0 ... N-1] = 1 };
static alignas(64) int8_t C41[16][48][256] = { [0 ... 16-1][0 ... 48-1][0 ... 256-1] = 1 };
static alignas(64) int8_t C42[16][48][256] = { [0 ... 16-1][0 ... 48-1][0 ... 256-1] = 1 };

static alignas(64) int8_t A5[12][16][N] = { [0 ... 12-1][0 ... 16-1][0 ... N-1] = 1 };
static alignas(64) int8_t B5[12][16][N] = { [0 ... 12-1][0 ... 16-1][0 ... N-1] = 1 };
static alignas(64) int8_t C51[16][16][256] = { [0 ... 16-1][0 ... 16-1][0 ... 256-1] = 1 };
static alignas(64) int8_t C52[16][16][256] = { [0 ... 16-1][0 ... 16-1][0 ... 256-1] = 1 };

static alignas(64) int8_t A6[1][784][N] = { [0 ... 1-1][0 ... 784-1][0 ... N-1] = 1 };
static alignas(64) int8_t B6[1][6][N] = { [0 ... 1-1][0 ... 6-1][0 ... N-1] = 1 };
static alignas(64) int8_t C61[784][6][256] = { [0 ... 784-1][0 ... 6-1][0 ... 256-1] = 1 };
static alignas(64) int8_t C62[784][6][256] = { [0 ... 784-1][0 ... 6-1][0 ... 256-1] = 1 };

static alignas(64) int8_t A7[1][196][N] = { [0 ... 1-1][0 ... 196-1][0 ... N-1] = 1 };
static alignas(64) int8_t B7[1][4][N] = { [0 ... 1-1][0 ... 4-1][0 ... N-1] = 1 };
static alignas(64) int8_t C71[196][4][256] = { [0 ... 196-1][0 ... 4-1][0 ... 256-1] = 1 };
static alignas(64) int8_t C72[196][4][256] = { [0 ... 196-1][0 ... 4-1][0 ... 256-1] = 1 };
static void whole_matmul(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 64 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 16 * sizeof(int8_t); //indicate the row size
    mcfgmi(16);
    mcfgki(64);
    mcfgni(16);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+6*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+6*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+7*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+7*16*stride_post, stride_post);

    mstb(0, (uint64_t *)start_addr_m0+0*16*stride, stride);
}
static void whole_matmul16(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 16 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 16 * sizeof(int8_t); //indicate the row size
    mcfgmi(16);
    mcfgki(16);
    mcfgni(16);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+6*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+6*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+7*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+7*16*stride_post, stride_post);

    mstb(0, (uint64_t *)start_addr_m0+0*16*stride, stride);
}
static void whole_madd(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 16 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 16 * sizeof(int8_t); //indicate the row size
    mcfgmi(16);
    mcfgki(16);
    mcfgni(16);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+6*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+6*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+7*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+7*16*stride_post, stride_post);

    mstb(0, (uint64_t *)start_addr_m0+0*16*stride, stride);
}

static void whole_matmul4(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 64 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 16 * sizeof(int8_t); //indicate the row size
    mcfgmi(16);
    mcfgki(64);
    mcfgni(16);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*16*stride_post, stride_post);
}
static void whole_matmul6(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 64 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 16 * sizeof(int8_t); //indicate the row size
    mcfgmi(16);
    mcfgki(16);
    mcfgni(16);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*16*stride_post, stride_post);
}
static void whole_madd4(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 16 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 16 * sizeof(int8_t); //indicate the row size
    mcfgmi(16);
    mcfgki(16);
    mcfgni(16);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*16*stride_post, stride_post);
}
static void whole_madd6(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 16 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 16 * sizeof(int8_t); //indicate the row size
    mcfgmi(16);
    mcfgki(16);
    mcfgni(16);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*16*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*16*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*16*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*16*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*16*stride_post, stride_post);
}
int main()
{
    
    for (int8_t k = 0; k < 8; k ++){
        for (int8_t i = 0; i < 16; i ++){
            for(int16_t j = 0; j < 1; j++){
                whole_matmul4(A1[k][i], B1[k][8*j], C11[i][8*j]);
            }
        }
    }

    for (int8_t k = 0; k < 7; k ++){
        for (int8_t i = 0; i < 16; i ++){
            for(int16_t j = 0; j < 1; j++){
                whole_madd4(C11[i][8*j], C12[i][8*j], C12[i][8*j]);
            }
        }
    }
    //=============
    // for (int16_t k = 0; k < 32; k ++){
    //     for (int8_t i = 0; i < 8; i ++){
    //         for(int16_t j = 0; j < 4; j++){
    //             whole_matmul(A2[k][i], B2[k][8*j], C21[i][8*j]);
    //         }
    //     }
    // }

    // for (int16_t k = 0; k < 31; k ++){
    //     for (int8_t i = 0; i < 8; i ++){
    //         for(int16_t j = 0; j < 4; j++){
    //             whole_madd(C21[i][8*j], C22[i][8*j], C21[i][8*j]);
    //         }
    //     }
    // }
    //==============
    // for (int8_t k = 0; k < 16; k ++){
    //     for (int16_t i = 0; i < 32; i ++){
    //         for(int16_t j = 0; j < 2; j++){
    //             whole_matmul(A3[k][i], B3[k][8*j], C31[i][8*j]);
    //         }
    //     }
    // }

    // for (int8_t k = 0; k < 15; k ++){
    //     for (int16_t i = 0; i < 32; i ++){
    //         for(int16_t j = 0; j < 2; j++){
    //             whole_madd(C31[i][8*j], C32[i][8*j], C32[i][8*j]);
    //         }
    //     }
    // }
    // /===============
    // for (int16_t k = 0; k < 48; k ++){
    //     for (int16_t i = 0; i < 16; i ++){
    //         for(int16_t j = 0; j < 6; j++){
    //             whole_matmul(A4[k][i], B4[k][8*j], C41[i][8*j]);
    //         }
    //     }
    // }

    // for (int16_t k = 0; k < 47; k ++){
    //     for (int16_t i = 0; i < 16; i ++){
    //         for(int16_t j = 0; j < 6; j++){
    //             whole_madd(C41[i][8*j], C42[i][8*j], C42[i][8*j]);
    //         }
    //     }
    // }
    //===============
    // for (int16_t k = 0; k < 12; k ++){
    //     for (int16_t i = 0; i < 16; i ++){
    //         for(int16_t j = 0; j < 2; j++){
    //             whole_matmul(A5[k][i], B5[k][8*j], C51[i][8*j]);
    //         }
    //     }
    // }

    // for (int16_t k = 0; k < 11; k ++){
    //     for (int16_t i = 0; i < 16; i ++){
    //         for(int16_t j = 0; j < 2; j++){
    //             whole_madd(C51[i][8*j], C52[i][8*j], C52[i][8*j]);
    //         }
    //     }
    // }
    //==========================
        // for (int16_t i = 0; i < 784; i ++){
        //     for(int16_t j = 0; j < 1; j++){
        //         whole_matmul6(A6[0][i], B6[0][8*j], C61[i][8*j]);
        //     }
        // }

    //========================
    // for (int16_t k = 0; k < 1; k ++){
    //     for (int16_t i = 0; i < 196; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             whole_matmul4(A7[k][i], B7[k][8*j], C71[i][8*j]);
    //         }
    //     }
    // }

    return 0;
}