/*
 * @Author: superboy
 * @Date: 2024-08-27 16:13:52
 * @LastEditTime: 2024-09-08 22:53:22
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/resnet50/conv3.c
 * 
 */
//======================
// input size: 56x56x256
//  最终矩阵乘法：128 x 256 和 256 x 3136, 128 x 1152 和 1152 x 3136, 512 x 128 和 128 x 3136
// 输出矩阵尺寸：64 x 12544
//======================
#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"
#define N 256

//等下乘以4
static alignas(32) int8_t A1[2][98][N] = {[0 ... 2-1][0 ... 98-1][0 ... N-1] = 1};
static alignas(32) int8_t B1[2][16][N] = {[0 ... 2-1][0 ... 16-1][0 ... N-1] = 1};
static alignas(32) int8_t C11[98][16][64] = {[0 ... 98-1][0 ... 16-1][0 ... 64-1] = 1};
static alignas(32) int8_t C12[98][16][64] = {[0 ... 98-1][0 ... 16-1][0 ... 64-1] = 1};

//dengxia chengyi 18
static alignas(32) int8_t A2[2][24][N] = {[0 ... 2-1][0 ... 24-1][0 ... N-1] = 1};
static alignas(32) int8_t B2[2][16][N] = {[0 ... 2-1][0 ... 16-1][0 ... N-1] = 1};
static alignas(32) int8_t C21[24][16][64] = {[0 ... 24-1][0 ... 16-1][0 ... 64-1] = 1};
static alignas(32) int8_t C22[24][16][64] = {[0 ... 24-1][0 ... 16-1][0 ... 64-1] = 1};

// cehngyi2
static alignas(32) int8_t A3[2][24][N] = {[0 ... 2-1][0 ... 16-1][0 ... N-1] = 1};
static alignas(32) int8_t B3[2][64][N] = {[0 ... 2-1][0 ... 64-1][0 ... N-1] = 1};
static alignas(32) int8_t C31[24][64][64] = {[0 ... 24-1][0 ... 64-1][0 ... 64-1] = 1};
static alignas(32) int8_t C32[24][64][64] = {[0 ... 24-1][0 ... 64-1][0 ... 64-1] = 1};

//cehngyi4
static alignas(32) int8_t A4[2][24][N] = {[0 ... 2-1][0 ... 16-1][0 ... N-1] = 1};
static alignas(32) int8_t B4[2][64][N] = {[0 ... 2-1][0 ... 64-1][0 ... N-1] = 1};
static alignas(32) int8_t C41[24][64][64] = {[0 ... 24-1][0 ... 64-1][0 ... 64-1] = 1};
static alignas(32) int8_t C42[24][64][64] = {[0 ... 24-1][0 ... 64-1][0 ... 64-1] = 1};

//chengyi9
static alignas(32) int8_t A5[2][24][N] = {[0 ... 2-1][0 ... 24-1][0 ... N-1] = 1};
static alignas(32) int8_t B5[2][16][N] = {[0 ... 2-1][0 ... 16-1][0 ... N-1] = 1};
static alignas(32) int8_t C51[24][16][64] = {[0 ... 24-1][0 ... 16-1][0 ... 64-1] = 1};
static alignas(32) int8_t C52[24][16][64] = {[0 ... 24-1][0 ... 16-1][0 ... 64-1] = 1};

//chengyi18
static alignas(32) int8_t A6[2][24][N] = {[0 ... 2-1][0 ... 24-1][0 ... N-1] = 1};
static alignas(32) int8_t B6[2][16][N] = {[0 ... 2-1][0 ... 16-1][0 ... N-1] = 1};
static alignas(32) int8_t C61[24][16][64] = {[0 ... 24-1][0 ... 16-1][0 ... 64-1] = 1};
static alignas(32) int8_t C62[24][16][64] = {[0 ... 24-1][0 ... 16-1][0 ... 64-1] = 1};

// cehngyi2
static alignas(32) int8_t A7[2][24][N] = {[0 ... 2-1][0 ... 16-1][0 ... N-1] = 1};
static alignas(32) int8_t B7[2][64][N] = {[0 ... 2-1][0 ... 64-1][0 ... N-1] = 1};
static alignas(32) int8_t C71[24][64][64] = {[0 ... 24-1][0 ... 64-1][0 ... 64-1] = 1};
static alignas(32) int8_t C72[24][64][64] = {[0 ... 24-1][0 ... 64-1][0 ... 64-1] = 1};

static alignas(32) int8_t C81[98][64][64] = {[0 ... 98-1][0 ... 64-1][0 ... 64-1] = 1};
static alignas(32) int8_t C82[98][64][64] = {[0 ... 98-1][0 ... 64-1][0 ... 64-1] = 1};
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
    // for (uint16_t i = 0; i < 98; i++){
    //     for (uint16_t j = 0; j < 2; j++){
    //         whole_matmul(A1[0][i], B1[0][8*j], C11[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 98; i++){
    //     for (uint16_t j = 0; j < 2; j++){
    //         whole_matmul(A1[1][i], B1[1][8*j], C12[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 98; i++){
    //     for (uint16_t j = 0; j < 2; j++){
    //         whole_madd(C11[i][8*j], C12[i][8*j], C11[i][8*j]);
    //     }
    // }
    //chengyi4
//#1
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 2; j++){
    //         whole_matmul(A2[0][i], B2[0][8*j], C21[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 2; j++){
    //         whole_matmul(A2[1][i], B2[1][8*j], C22[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 2; j++){
    //         whole_madd(C21[i][8*j], C22[i][8*j], C21[i][8*j]);
    //     }
    // }
    //以上乘以18
//#2
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A3[0][i], B3[0][8*j], C31[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A3[1][i], B3[1][8*j], C32[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_madd(C31[i][8*j], C32[i][8*j], C31[i][8*j]);
    //     }
    // }
    //chebngyi2
//#3
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A4[0][i], B4[0][8*j], C41[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A4[1][i], B4[1][8*j], C42[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_madd(C41[i][8*j], C42[i][8*j], C41[i][8*j]);
    //     }
    // }
//chengyi4
//#4=========================================
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 2; j++){
    //         whole_matmul(A5[0][i], B5[0][8*j], C51[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 2; j++){
    //         whole_matmul(A5[1][i], B5[1][8*j], C52[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 2; j++){
    //         whole_madd(C51[i][8*j], C52[i][8*j], C51[i][8*j]);
    //     }
    // }
    //以上乘以9
//#5
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 2; j++){
    //         whole_matmul(A6[0][i], B6[0][8*j], C61[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 2; j++){
    //         whole_matmul(A6[1][i], B6[1][8*j], C62[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 2; j++){
    //         whole_madd(C61[i][8*j], C62[i][8*j], C61[i][8*j]);
    //     }
    // }
    //以上乘以18
//#6
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A7[0][i], B7[0][8*j], C71[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_matmul(A7[1][i], B7[1][8*j], C72[i][8*j]);
    //     }
    // }
    // for (uint16_t i = 0; i < 24; i++){
    //     for (uint16_t j = 0; j < 8; j++){
    //         whole_madd(C71[i][8*j], C72[i][8*j], C71[i][8*j]);
    //     }
    // }
    //以上乘以2
//#7===============================以上乘以3
    //算residual
    for (uint8_t i = 0; i < 98; i++){
        for (uint8_t j = 0; j < 8; j++){
            whole_madd(C81[i][8*j], C82[i][8*j], C81[i][8*j]);
        }
    }
    //以上乘以4
  //#8
    return 0;
}