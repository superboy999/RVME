/*
 * @Author: superboy
 * @Date: 2024-09-08 22:08:46
 * @LastEditTime: 2024-09-22 04:03:12
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/exploration/128.c
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
#define N 64 //indicate the whole matrix length

static alignas(16) int8_t A1[32][64][N] = { [0 ... 32-1][0 ... 64-1][0 ... N-1] = 1 };
static alignas(16) int8_t B1[32][16][N] = { [0 ... 32-1][0 ... 16-1][0 ... N-1] = 1 };
static alignas(16) int8_t C11[64][16][16] = { [0 ... 64-1][0 ... 16-1][0 ... 16-1] = 1 };
static alignas(16) int8_t C12[64][16][16] = { [0 ... 64-1][0 ... 16-1][0 ... 16-1] = 1 };

static alignas(16) int8_t A2[128][32][N] = { [0 ... 128-1][0 ... 32-1][0 ... N-1] = 1 };
static alignas(16) int8_t B2[128][128][N] = { [0 ... 128-1][0 ... 128-1][0 ... N-1] = 1 };
static alignas(16) int8_t C21[32][128][16] = { [0 ... 32-1][0 ... 128-1][0 ... 16-1] = 1 };
static alignas(16) int8_t C22[32][128][16] = { [0 ... 32-1][0 ... 128-1][0 ... 16-1] = 1 };

static alignas(16) int8_t A3[64][128][N] = { [0 ... 64-1][0 ... 128-1][0 ... N-1] = 1 };
static alignas(16) int8_t B3[64][64][N] = { [0 ... 64-1][0 ... 64-1][0 ... N-1] = 1 };
static alignas(16) int8_t C31[128][64][16] = { [0 ... 128-1][0 ... 64-1][0 ... 16-1] = 1 };
static alignas(16) int8_t C32[128][64][16] = { [0 ... 128-1][0 ... 64-1][0 ... 16-1] = 1 };

static alignas(64) int8_t A4[192][64][N] = { [0 ... 192-1][0 ... 64-1][0 ... N-1] = 1 };
static alignas(64) int8_t B4[192][192][N] = { [0 ... 192-1][0 ... 192-1][0 ... N-1] = 1 };
static alignas(64) int8_t C41[64][192][16] = { [0 ... 64-1][0 ... 192-1][0 ... 16-1] = 1 };
static alignas(64) int8_t C42[64][192][16] = { [0 ... 64-1][0 ... 192-1][0 ... 16-1] = 1 };

static alignas(64) int8_t A5[48][64][N] = { [0 ... 48-1][0 ... 64-1][0 ... N-1] = 1 };
static alignas(64) int8_t B5[48][64][N] = { [0 ... 48-1][0 ... 64-1][0 ... N-1] = 1 };
static alignas(64) int8_t C51[64][64][16] = { [0 ... 64-1][0 ... 64-1][0 ... 16-1] = 1 };
static alignas(64) int8_t C52[64][64][16] = { [0 ... 64-1][0 ... 64-1][0 ... 16-1] = 1 };

static alignas(64) int8_t A6[3136][N] = { [0 ... 3136-1][0 ... N-1] = 1 };
static alignas(64) int8_t B6[24][N] = { [0 ... 24-1][0 ... N-1] = 1 };
static alignas(64) int8_t C61[3136][24][16] = { [0 ... 3136-1][0 ... 24-1][0 ... 16-1] = 1 };
static alignas(64) int8_t C62[3136][24][16] = { [0 ... 3136-1][0 ... 24-1][0 ... 16-1] = 1 };

static alignas(64) int8_t A7[4][784][N] = { [0 ... 4-1][0 ... 784-1][0 ... N-1] = 1 };
static alignas(64) int8_t B7[4][16][N] = { [0 ... 4-1][0 ... 16-1][0 ... N-1] = 1 };
static alignas(64) int8_t C71[784][16][16] = { [0 ... 784-1][0 ... 16-1][0 ... 16-1] = 1 };
static alignas(64) int8_t C72[784][16][16] = { [0 ... 784-1][0 ... 16-1][0 ... 16-1] = 1 };

static void whole_matmul(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 16 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 4 * sizeof(int8_t); //indicate the row size
    mcfgmi(4);
    mcfgki(16);
    mcfgni(4);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*4*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*4*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*4*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+6*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+6*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+7*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+7*4*stride_post, stride_post);

    mstb(0, (uint64_t *)start_addr_m0+0*4*stride, stride);
}
static void whole_madd(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 16 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 4 * sizeof(int8_t); //indicate the row size
    mcfgmi(4);
    mcfgki(4);
    mcfgni(4);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*4*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*4*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*4*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+6*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+6*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+7*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+7*4*stride_post, stride_post);

    mstb(0, (uint64_t *)start_addr_m0+0*4*stride, stride);
}

static void whole_matmul6(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 16 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 4 * sizeof(int8_t); //indicate the row size
    mcfgmi(4);
    mcfgki(16);
    mcfgni(4);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*4*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*4*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*4*stride_post, stride_post);
}
static void whole_madd6(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 16 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 4 * sizeof(int8_t); //indicate the row size
    mcfgmi(4);
    mcfgki(4);
    mcfgni(4);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*4*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+2*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+3*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*4*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*4*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*4*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*4*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*4*stride_post, stride_post);
}

int main()
{
    
    for (int8_t k = 0; k < 32; k ++){
        for (int8_t i = 0; i < 64; i ++){
            for(int16_t j = 0; j < 2; j++){
                whole_matmul(A1[k][i], B1[k][8*j], C11[i][8*j]);
            }
        }
    }

    for (int8_t k = 0; k < 31; k ++){
        for (int8_t i = 0; i < 64; i ++){
            for(int16_t j = 0; j < 2; j++){
                whole_madd(C11[i][8*j], C12[i][8*j], C11[i][8*j]);
            }
        }
    }
    //=============
    // for (int16_t k = 0; k < 128; k ++){
    //     for (int8_t i = 0; i < 32; i ++){
    //         for(int16_t j = 0; j < 16; j++){
    //             whole_matmul(A2[k][i], B2[k][8*j], C21[i][8*j]);
    //         }
    //     }
    // }

    // for (int16_t k = 0; k < 127; k ++){
    //     for (int8_t i = 0; i < 32; i ++){
    //         for(int16_t j = 0; j < 16; j++){
    //             whole_madd(C21[i][8*j], C22[i][8*j], C21[i][8*j]);
    //         }
    //     }
    // }
    //==============
    // for (int8_t k = 0; k < 64; k ++){
    //     for (int16_t i = 0; i < 128; i ++){
    //         for(int16_t j = 0; j < 8; j++){
    //             whole_matmul(A3[k][i], B3[k][8*j], C31[i][8*j]);
    //         }
    //     }
    // }

    // for (int8_t k = 0; k < 63; k ++){
    //     for (int16_t i = 0; i < 128; i ++){
    //         for(int16_t j = 0; j < 8; j++){
    //             whole_madd(C31[i][8*j], C32[i][8*j], C31[i][8*j]);
    //         }
    //     }
    // }
    // /===============
    // for (int16_t k = 0; k < 192; k ++){
    //     for (int16_t i = 0; i < 64; i ++){
    //         for(int16_t j = 0; j < 24; j++){
    //             whole_matmul(A4[k][i], B4[k][8*j], C41[i][8*j]);
    //         }
    //     }
    // }

    // for (int16_t k = 0; k < 191; k ++){
    //     for (int16_t i = 0; i < 64; i ++){
    //         for(int16_t j = 0; j < 24; j++){
    //             whole_madd(C41[i][8*j], C42[i][8*j], C42[i][8*j]);
    //         }
    //     }
    // }
    //===============
    // for (int8_t k = 0; k < 48; k ++){
    //     for (int8_t i = 0; i < 64; i ++){
    //         for(int8_t j = 0; j < 8; j++){
    //             whole_matmul(A5[k][i], B5[k][8*j], C52[i][8*j]);
    //         }
    //     }
    // }

    // for (int8_t k = 0; k < 47; k ++){
    //     for (int8_t i = 0; i < 64; i ++){
    //         for(int8_t j = 0; j < 8; j++){
    //             whole_madd(C52[i][8*j], C51[i][8*j], C52[i][8*j]);
    //         }
    //     }
    // }
    //==========================
    // for (int16_t i = 0; i < 3136; i ++){
    //     for(int16_t j = 0; j < 4; j++){
    //         whole_matmul6(A6[i], B6[8*j], C61[i][8*j]);
    //     }
    // }
    // for (int16_t i = 0; i < 3136; i ++){
    //     for(int16_t j = 0; j < 4; j++){
    //         whole_madd6(C61[i][8*j], C62[i][8*j], C62[i][8*j]);
    //     }
    // }
    //========================
    // for (int16_t k = 0; k < 4; k ++){
    //     for (int16_t i = 0; i < 784; i ++){
    //         for(int16_t j = 0; j < 2; j++){
    //             whole_matmul(A7[k][i], B7[k][8*j], C71[i][8*j]);
    //         }
    //     }
    // }
    // for (int16_t i = 0; i < 784; i ++){
    //     for(int16_t j = 0; j < 2; j++){
    //         whole_madd(C71[i][8*j], C72[i][8*j], C71[i][8*j]);
    //     }
    // }
    // for (int16_t i = 0; i < 784; i ++){
    //     for(int16_t j = 0; j < 2; j++){
    //         whole_madd(C71[i][8*j], C72[i][8*j], C71[i][8*j]);
    //     }
    // }
    // for (int16_t i = 0; i < 784; i ++){
    //     for(int16_t j = 0; j < 2; j++){
    //         whole_madd(C71[i][8*j], C72[i][8*j], C71[i][8*j]);
    //     }
    // }
    return 0;
}