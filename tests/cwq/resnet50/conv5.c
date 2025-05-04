/*
 * @Author: superboy
 * @Date: 2024-08-27 16:14:58
 * @LastEditTime: 2024-09-07 01:53:14
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/resnet50/conv5.c
 * 
 */
#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"
#define N 256
//======================
// input size: 56x56x256
//  最终矩阵乘法：'512x1024'和'1024 x 196, `512x4608`和`4608 x196, '2048 x512'和'512 x 196
// 输出矩阵尺寸：64 x 12544
//======================

//等下乘以16
static alignas(32) int8_t A1[2][6][N] = {[0 ... 2-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t B1[2][64][N] = {[0 ... 2-1][0 ... 64-1][0 ... N-1] = 1};
static alignas(32) int8_t C11[6][64][64] = {[0 ... 6-1][0 ... 64-1][0 ... 64-1] = 1};
static alignas(32) int8_t C12[6][64][64] = {[0 ... 6-1][0 ... 64-1][0 ... 64-1] = 1};

//dengxia chengyi 72
static alignas(32) int8_t A2[2][6][N] = {[0 ... 2-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t B2[2][64][N] = {[0 ... 2-1][0 ... 64-1][0 ... N-1] = 1};
static alignas(32) int8_t C21[6][64][64] = {[0 ... 6-1][0 ... 64-1][0 ... 64-1] = 1};
static alignas(32) int8_t C22[6][64][64] = {[0 ... 6-1][0 ... 64-1][0 ... 64-1] = 1};

// cehngyi8
static alignas(32) int8_t A3[2][6][N] = {[0 ... 2-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t B3[2][256][N] = {[0 ... 2-1][0 ... 256-1][0 ... N-1] = 1};
static alignas(32) int8_t C31[6][256][64] = {[0 ... 6-1][0 ... 256-1][0 ... 64-1] = 1};
static alignas(32) int8_t C32[6][256][64] = {[0 ... 6-1][0 ... 256-1][0 ... 64-1] = 1};

//cehngyi16
static alignas(32) int8_t A4[2][6][N] = {[0 ... 2-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t B4[2][256][N] = {[0 ... 2-1][0 ... 256-1][0 ... N-1] = 1};
static alignas(32) int8_t C41[6][256][64] = {[0 ... 6-1][0 ... 256-1][0 ... 64-1] = 1};
static alignas(32) int8_t C42[6][256][64] = {[0 ... 6-1][0 ... 256-1][0 ... 64-1] = 1};

//chengyi32
static alignas(32) int8_t A5[2][6][N] = {[0 ... 2-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t B5[2][64][N] = {[0 ... 2-1][0 ... 64-1][0 ... N-1] = 1};
static alignas(32) int8_t C51[6][64][64] = {[0 ... 6-1][0 ... 64-1][0 ... 64-1] = 1};
static alignas(32) int8_t C52[6][64][64] = {[0 ... 6-1][0 ... 64-1][0 ... 64-1] = 1};

//chengyi 72
static alignas(32) int8_t A6[2][6][N] = {[0 ... 2-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t B6[2][64][N] = {[0 ... 2-1][0 ... 64-1][0 ... N-1] = 1};
static alignas(32) int8_t C61[6][64][64] = {[0 ... 6-1][0 ... 64-1][0 ... 64-1] = 1};
static alignas(32) int8_t C62[6][64][64] = {[0 ... 6-1][0 ... 64-1][0 ... 64-1] = 1};

// cehngyi8
static alignas(32) int8_t A7[2][6][N] = {[0 ... 2-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t B7[2][256][N] = {[0 ... 2-1][0 ... 256-1][0 ... N-1] = 1};
static alignas(32) int8_t C71[6][256][64] = {[0 ... 6-1][0 ... 256-1][0 ... 64-1] = 1};
static alignas(32) int8_t C72[6][256][64] = {[0 ... 6-1][0 ... 256-1][0 ... 64-1] = 1};

static alignas(32) int8_t C81[6][256][64] = {[0 ... 6-1][0 ... 256-1][0 ... 64-1] = 1};
static alignas(32) int8_t C82[6][256][64] = {[0 ... 6-1][0 ... 256-1][0 ... 64-1] = 1};

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

int main()
{
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A1[0][i], B1[0][8*j], C11[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A1[0][i], B1[0][8*j], C12[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_madd(C11[i][8*j], C12[i][8*j], C11[i][8*j]);
    //     }
    // }
    //chengyi16
//#1
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A2[0][i], B2[0][8*j], C21[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A2[0][i], B2[0][8*j], C22[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_madd(C21[i][8*j], C22[i][8*j], C21[i][8*j]);
    //     }
    // }
    //以上乘以72
//#2
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 32; j++){
    //         whole_matmul(A3[0][i], B3[0][8*j], C31[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 32; j++){
    //         whole_matmul(A3[0][i], B3[0][8*j], C32[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 32; j++){
    //         whole_madd(C31[i][8*j], C32[i][8*j], C31[i][8*j]);
    //     }
    // }
    //chebngyi8
//#3
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 32; j++){
    //         whole_matmul(A4[0][i], B4[0][8*j], C41[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 32; j++){
    //         whole_matmul(A4[0][i], B4[0][8*j], C42[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 32; j++){
    //         whole_madd(C41[i][8*j], C42[i][8*j], C41[i][8*j]);
    //     }
    // }
//chengyi16
//#4=========================================
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A5[0][i], B5[0][8*j], C51[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A5[0][i], B5[0][8*j], C52[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_madd(C51[i][8*j], C52[i][8*j], C51[i][8*j]);
    //     }
    // }
    //以上乘以32
//#5
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A6[0][i], B6[0][8*j], C61[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A6[0][i], B6[0][8*j], C62[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_madd(C61[i][8*j], C62[i][8*j], C61[i][8*j]);
    //     }
    // }
    //以上乘以72
//#6
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 32; j++){
    //         whole_matmul(A7[0][i], B7[0][8*j], C71[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 32; j++){
    //         whole_matmul(A7[0][i], B7[0][8*j], C72[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 6; i++){
    //     for (uint16_t j = 0; j < 32; j++){
    //         whole_madd(C71[i][8*j], C72[i][8*j], C71[i][8*j]);
    //     }
    // }
    //以上乘以8
//#7===============================以上乘以2
    //算residual
    for (uint8_t i = 0; i < 6; i++){
        for (uint8_t j = 0; j < 32; j++){
            whole_madd(C81[i][8*j], C82[i][8*j], C81[i][8*j]);
        }
    }
    //以上乘以3
  //#8
    return 0;
}