/*
 * @Author: superboy
 * @Date: 2024-08-23 14:36:56
 * @LastEditTime: 2024-09-05 23:59:44
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/transformer/embeding.c
 * 
 */
//====================
// size description
// token 256*32000
// dmodel=512
// 输出的是256*512
// ===================

#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"
#define N 256 //indicate the whole matrix elements

static alignas(32) int8_t INPUT[2][32][N] = { [0 ... 2-1][0 ... 32-1][0 ... N-1] = 1 };
static alignas(32) int8_t WE[2][64][N] = { [0 ... 2-1][0 ... 64-1][0 ... N-1] = 1 };
static alignas(32) int8_t EM_VEC1[33][65][64] = { [0 ... 33-1][0 ... 65-1][0 ... 64-1] = 1 };
static alignas(32) int8_t EM_VEC2[33][65][64] = { [0 ... 33-1][0 ... 65-1][0 ... 64-1] = 1 };

// static void whole_matmul(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
// {
//     //continuly compute 8
//     uint64_t stride = 32 * sizeof(int8_t); //indicate the row size
//     mcfgmi(8);
//     mcfgki(32);
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
// static void whole_madd(uint64_t *start_addr_A, uint64_t *start_addr_B, uint64_t *start_addr_C)
// {
//     //compute 8 times
//     uint64_t stride = 8 * sizeof(int8_t); //indicate the row size
//     mcfgmi(8);
//     mcfgki(8);
//     mcfgni(8);
//     mldb_m2((uint64_t *)start_addr_A+0*4*8*stride, stride);
//     mldb_m3((uint64_t *)start_addr_B+0*4*8*stride, stride);
//     mldb_m4((uint64_t *)start_addr_A+1*4*8*stride, stride);
//     mldb_m5((uint64_t *)start_addr_B+1*4*8*stride, stride);
//     mldb_m6((uint64_t *)start_addr_A+2*4*8*stride, stride);
//     mldb_m7((uint64_t *)start_addr_B+2*4*8*stride, stride);
//     madd(0, 2, 3);
//     mstb(0, (uint64_t *)start_addr_C+0*4*8*stride, stride);
//     madd(1, 4, 5);
//     mstb(1, (uint64_t *)start_addr_C+1*4*8*stride, stride);
//     madd(0, 6, 7);
//     mstb(0, (uint64_t *)start_addr_C+2*4*8*stride, stride);
//     mldb_m2((uint64_t *)start_addr_A+3*4*8*stride, stride);
//     mldb_m3((uint64_t *)start_addr_B+3*4*8*stride, stride);
//     mldb_m4((uint64_t *)start_addr_A+4*4*8*stride, stride);
//     mldb_m5((uint64_t *)start_addr_B+4*4*8*stride, stride);
//     mldb_m6((uint64_t *)start_addr_A+5*4*8*stride, stride);
//     mldb_m7((uint64_t *)start_addr_B+5*4*8*stride, stride);
//     madd(1, 2, 3);
//     mstb(1, (uint64_t *)start_addr_C+3*4*8*stride, stride);
//     madd(0, 4, 5);
//     mstb(0, (uint64_t *)start_addr_C+4*4*8*stride, stride);
//     madd(1, 6, 7);
//     mstb(1, (uint64_t *)start_addr_C+5*4*8*stride, stride);

//     mldb_m2((uint64_t *)start_addr_A+0*4*8*stride, stride);
//     mldb_m3((uint64_t *)start_addr_B+0*4*8*stride, stride);
//     mldb_m4((uint64_t *)start_addr_A+1*4*8*stride, stride);
//     mldb_m5((uint64_t *)start_addr_B+1*4*8*stride, stride);
//     madd(0, 2, 3);
//     mstb(0, (uint64_t *)start_addr_C+0*4*8*stride, stride);
//     madd(1, 4, 5);
//     mstb(1, (uint64_t *)start_addr_C+1*4*8*stride, stride);
// }

static void matmul(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
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

static void madd_8(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
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
    // for(uint8_t i = 0; i < 32; i++){
    //     for(uint8_t j = 0; j < 8; j++){
    //         matmul(INPUT[0][i], WE[0][j*8], EM_VEC1[i][j*8]);
    //     }
    // }

    // for(uint8_t i = 0; i < 32; i++){
    //     for(uint8_t j = 0; j < 8; j++){
    //         matmul(INPUT[1][i], WE[1][j*8], EM_VEC2[i][j*8]);
    //     }
    // }

    // for(uint8_t i = 0; i < 32; i++){
    //     for(uint8_t j = 0; j < 8; j++){
    //         madd_8(EM_VEC1[i][j*8], EM_VEC2[i][j*8], EM_VEC1[i][j*8]);
    //     }
    // }
//#1
    //以上的时间乘以1000/2，就是完整的计算出嵌入向量的周期
    //以下计算加位置编码的计算量

    for(uint8_t i = 0; i < 32; i++){
        for(uint8_t j = 0; j < 8; j++){
            madd_8(EM_VEC1[i][j*8], EM_VEC2[i][j*8], EM_VEC1[i][j*8]);
        }
    }
//#2
    return 0;
}
