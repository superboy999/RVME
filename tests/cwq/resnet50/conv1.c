/*
 * @Author: superboy
 * @Date: 2024-08-27 16:13:37
 * @LastEditTime: 2025-03-13 17:23:09
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/resnet50/conv1.c
 * 
 */
#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"
//======================
// input size: 224*224*3
//  最终矩阵乘法：64 x 147 和 147 x 12544
// 输出矩阵尺寸：64 x 12544
//======================

#define N 256
static alignas(32) int8_t A1[9][8][N] = { [0 ... 9-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t A2[2][9][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t B1[9][1568][N] = { [0 ... 9-1][0 ... 1568-1][0 ... N-1] = 1};
static alignas(32) int8_t B2[2][1568][N] = {[0 ... 2-1][0 ... 1568-1][0 ... N-1] = 1};
static alignas(32) int8_t C1[9][1568][64] = {[0 ... 8-1][0 ... 1568-1][0 ... 64-1] = 1};
static alignas(32) int8_t C2[9][1568][64] = {[0 ... 8-1][0 ... 1568-1][0 ... 64-1] = 1};
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

static void tail_matmul(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size
    uint64_t stride_1 = 19 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(19);
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
}

// static void tail_matmul(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
// {
//     //continuly compute 8
//     uint64_t stride = 19 * sizeof(int8_t); //indicate the row size
//     mcfgmi(8);
//     mcfgki(19);
//     mcfgni(8);
//     mldb_m0((uint64_t *)start_addr_m0, stride);
//     mldb_m4((uint64_t *)start_addr_B+0*4*8*stride, stride);
//     mldb_m5((uint64_t *)start_addr_B+1*4*8*stride, stride);
//     mldb_m6((uint64_t *)start_addr_B+2*4*8*stride, stride);
//     mldb_m7((uint64_t *)start_addr_B+3*4*8*stride, stride);
//     matrixmul_int8_ss(1, 0, 4);
//     mstb(1, (uint64_t *)start_addr_C+0*4*8*stride, stride);
//     matrixmul_int8_ss(2, 0, 5);
//     mstb(2, (uint64_t *)start_addr_C+1*4*8*stride, stride);
//     matrixmul_int8_ss(3, 0, 6);
//     mstb(3, (uint64_t *)start_addr_C+2*4*8*stride, stride);
//     matrixmul_int8_ss(1, 0, 7);
//     mstb(4, (uint64_t *)start_addr_C+3*4*8*stride, stride);
//     mldb_m4((uint64_t *)start_addr_B+4*4*8*stride, stride);
//     mldb_m5((uint64_t *)start_addr_B+5*4*8*stride, stride);
//     mldb_m6((uint64_t *)start_addr_B+6*4*8*stride, stride);
//     mldb_m7((uint64_t *)start_addr_B+7*4*8*stride, stride);
//     matrixmul_int8_ss(2, 0, 4);
//     mstb(2, (uint64_t *)start_addr_C+4*4*8*stride, stride);
//     matrixmul_int8_ss(3, 0, 5);
//     mstb(3, (uint64_t *)start_addr_C+5*4*8*stride, stride);
//     matrixmul_int8_ss(1, 0, 6);
//     mstb(1, (uint64_t *)start_addr_C+6*4*8*stride, stride);
//     matrixmul_int8_ss(2, 0, 7);
//     mstb(2, (uint64_t *)start_addr_C+7*4*8*stride, stride);

//     mstb(0, (uint64_t *)start_addr_m0+1*4*8*stride, stride);
// }

int main()
{
    // for (uint8_t k = 0; k < 9; k++){
    //     for (uint8_t i = 0; i < 8; i++){
    //         for (uint16_t j = 0; j < 196; j++){
    //             whole_matmul(A1[k][i], B1[k][8*j], C1[i][8*j]);
    //         }
    //     }
    // }

    // for (uint8_t k = 0; k < 8; k++){
    //     for (uint8_t i = 0; i < 8; i++){
    //         for (uint16_t j = 0; j < 196; j++){
    //             whole_madd(C1[i][8*j], C2[i][8*j], C1[i][8*j]);
    //         }
    //     }
    // }
//#1
    //以上乘以2代表了所有能被完整映射的矩阵乘完的结果
    // for (uint8_t i = 0; i < 8; i++){
    //     for (uint16_t j = 0; j < 196; j++){
    //         tail_matmul(A2[0][i], B2[0][8*j], C1[i][8*j]);
    //     }
    // }   

    // for (uint8_t i = 0; i < 8; i++){
    //     for (uint16_t j = 0; j < 196; j++){
    //         whole_madd(C1[i][8*j], C2[i][8*j], C1[i][8*j]);
    //     }
    // }
//#2
    return 0;
}