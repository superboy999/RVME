/*
 * @Author: superboy
 * @Date: 2024-08-27 16:13:29
 * @LastEditTime: 2024-09-07 01:19:07
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/resnet50/conv2.c
 * 
 */
//======================
// input size: 112x112x64
//  最终矩阵乘法：64 x 64 和 64 x 12544, 64 x 576和576 x 12544, 256 x 64 和 64 x 12544
// 输出矩阵尺寸：256 x 12544
//======================
#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"
#define N 256

static alignas(32) int8_t A1[2][392][N] = {[0 ... 2-1][0 ... 392-1][0 ... N-1] = 1};
static alignas(32) int8_t B1[2][8][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t C11[392][8][64] = {[0 ... 392-1][0 ... 8-1][0 ... 64-1] = 1};
static alignas(32) int8_t C12[392][8][64] = {[0 ... 392-1][0 ... 8-1][0 ... 64-1] = 1};

//等会乘以9
static alignas(32) int8_t A2[2][392][N] = {[0 ... 2-1][0 ... 392-1][0 ... N-1] = 1};
static alignas(32) int8_t B2[2][8][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t C21[392][8][64] = {[0 ... 392-1][0 ... 8-1][0 ... 64-1] = 1};
static alignas(32) int8_t C22[392][8][64] = {[0 ... 392-1][0 ... 8-1][0 ... 64-1] = 1};

static alignas(32) int8_t A3[2][392][N] = {[0 ... 2-1][0 ... 392-1][0 ... N-1] = 1};
static alignas(32) int8_t B3[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t C31[392][32][N] = {[0 ... 392-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t C32[392][32][N] = {[0 ... 392-1][0 ... 32-1][0 ... N-1] = 1};

static alignas(32) int8_t A4[2][392][N] = {[0 ... 2-1][0 ... 392-1][0 ... N-1] = 1};
static alignas(32) int8_t B4[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t C41[392][32][N] = {[0 ... 392-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t C42[392][32][N] = {[0 ... 392-1][0 ... 32-1][0 ... N-1] = 1};

//等会乘以4
static alignas(32) int8_t A5[2][392][N] = {[0 ... 2-1][0 ... 392-1][0 ... N-1] = 1};
static alignas(32) int8_t B5[2][8][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t C51[392][8][N] = {[0 ... 392-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t C52[392][8][N] = {[0 ... 392-1][0 ... 8-1][0 ... N-1] = 1};

static alignas(32) int8_t A6[2][392][N] = {[0 ... 2-1][0 ... 392-1][0 ... N-1] = 1};
static alignas(32) int8_t B6[2][8][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t C61[392][8][64] = {[0 ... 392-1][0 ... 8-1][0 ... 64-1] = 1};
static alignas(32) int8_t C62[392][8][64] = {[0 ... 392-1][0 ... 8-1][0 ... 64-1] = 1};

static alignas(32) int8_t A7[2][392][N] = {[0 ... 2-1][0 ... 392-1][0 ... N-1] = 1};
static alignas(32) int8_t B7[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t C71[392][32][N] = {[0 ... 392-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t C72[392][32][N] = {[0 ... 392-1][0 ... 32-1][0 ... N-1] = 1};
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
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 1; j++){
    //         whole_matmul(A1[0][i], B1[0][8*j], C11[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 1; j++){
    //         whole_matmul(A1[1][i], B1[1][8*j], C12[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 1; j++){
    //         whole_madd(C11[i][8*j], C12[i][8*j], C11[i][8*j]);
    //     }
    // }
//#1
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 1; j++){
    //         whole_matmul(A2[0][i], B2[0][8*j], C21[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 1; j++){
    //         whole_matmul(A2[1][i], B2[1][8*j], C22[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 1; j++){
    //         whole_madd(C21[i][8*j], C22[i][8*j], C21[i][8*j]);
    //     }
    // }
    //以上乘以9
//#2
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 4; j++){
    //         whole_matmul(A3[0][i], B3[0][8*j], C31[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 4; j++){
    //         whole_matmul(A3[1][i], B3[1][8*j], C32[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 4; j++){
    //         whole_madd(C31[i][8*j], C32[i][8*j], C31[i][8*j]);
    //     }
    // }
//#3
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 4; j++){
    //         whole_matmul(A4[0][i], B4[0][8*j], C41[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 4; j++){
    //         whole_matmul(A4[1][i], B4[1][8*j], C42[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 4; j++){
    //         whole_madd(C41[i][8*j], C42[i][8*j], C41[i][8*j]);
    //     }
    // }

//#4====================
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 1; j++){
    //         whole_matmul(A5[0][i], B5[0][8*j], C51[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 1; j++){
    //         whole_matmul(A5[0][i], B5[0][8*j], C52[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 1; j++){
    //         whole_madd(C51[i][8*j], C52[i][8*j], C51[i][8*j]);
    //     }
    // }
    //以上乘以4
//#5
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 1; j++){
    //         whole_matmul(A6[0][i], B6[0][8*j], C61[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 1; j++){
    //         whole_matmul(A6[1][i], B6[1][8*j], C62[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 1; j++){
    //         whole_madd(C61[i][8*j], C62[i][8*j], C61[i][8*j]);
    //     }
    // }
//#6
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 4; j++){
    //         whole_matmul(A7[0][i], B7[0][8*j], C71[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 4; j++){
    //         whole_matmul(A7[1][i], B7[1][8*j], C72[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 4; j++){
    //         whole_madd(C71[i][8*j], C72[i][8*j], C71[i][8*j]);
    //     }
    // }
//#7 ========================= 以上乘以2
    //以下算residual
    // for (uint8_t i = 0; i < 392; i++){
    //     for (uint16_t j = 0; j < 4; j++){
    //         whole_madd(C31[i][8*j], C32[i][8*j], C31[i][8*j]);
    //     }
    // }
    //以上乘以3
    return 0;
//#8

}