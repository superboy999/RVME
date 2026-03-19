/*
 * @Author: superboy
 * @Date: 2024-08-19 13:37:55
 * @LastEditTime: 2024-09-03 21:31:05
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/transformer_test/test.c
 * 
 */
#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"

#define N 256 //indicate the whole matrix elements

static alignas(32) int8_t token1[32][N] = {[0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t WQ1[8][N] = {[0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t token2[32][N] = {[0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t WQ2[8][N] = {[0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t Q1[32][8][64] = {[0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1};
static alignas(32) int8_t Q2[32][8][64] = {[0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1};
static alignas(32) int8_t Q3[32][8][64] = {[0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1};


// static alignas(32) int8_t QQ1[2][8][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
// static alignas(32) int8_t K1[2][8][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
// static alignas(32) int8_t QKT1[32][32][64] = {[0 ... 32-1][0 ... 32-1][0 ... 64-1] = 1};
// static alignas(32) int8_t QKT2[32][32][64] = {[0 ... 32-1][0 ... 32-1][0 ... 64-1] = 1};

// static alignas(32) int8_t QK1[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
// static alignas(32) int8_t V[2][8][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
// static alignas(32) int8_t QKV1[32][8][64] = {[0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1};
// static alignas(32) int8_t QKV2[32][8][64] = {[0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1};

// static alignas(32) int8_t CONC[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
// static alignas(32) int8_t WFC[2][64][N] = {[0 ... 2-1][0 ... 64-1][0 ... N-1] = 1};
// static alignas(32) int8_t CONC1[32][64][64] = {[0 ... 32-1][0 ... 64-1][0 ... 64-1] = 1};
// static alignas(32) int8_t CONC2[32][64][64] = {[0 ... 32-1][0 ... 64-1][0 ... 64-1] = 1};

// static alignas(32) int8_t FF_IN[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
// static alignas(32) int8_t WFF1[2][256][N] = {[0 ... 2-1][0 ... 256-1][0 ... N-1] = 1};
// static alignas(32) int8_t WFFB1[32][256][64] = {[0 ... 2-1][0 ... 256-1][0 ... 64-1] = 1};

// static alignas(32) int8_t WA[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
// static alignas(32) int8_t WB[2][64][N] = {[0 ... 2-1][0 ... 64-1][0 ... N-1] = 1};
// static alignas(32) int8_t WC1[32][64][64] = {[0 ... 32-1][0 ... 64-1][0 ... 64-1] = 1};
// static alignas(32) int8_t WC2[32][64][64] = {[0 ... 32-1][0 ... 64-1][0 ... 64-1] = 1};

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
    mldb_m5((uint64_t *)start_addr_B+1*8*stride, stride);
    mldb_m6((uint64_t *)start_addr_B+2*8*stride, stride);
    mldb_m7((uint64_t *)start_addr_B+3*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*8*stride_post, stride_post);
    matrixmul_int8_ss(2, 0, 5);
    mstb(2, (uint64_t *)start_addr_C+1*8*stride_post, stride_post);
    matrixmul_int8_ss(3, 0, 6);
    mstb(3, (uint64_t *)start_addr_C+2*8*stride_post, stride_post);
    matrixmul_int8_ss(1, 0, 7);
    mstb(4, (uint64_t *)start_addr_C+3*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+5*8*stride, stride);
    mldb_m6((uint64_t *)start_addr_B+6*8*stride, stride);
    mldb_m7((uint64_t *)start_addr_B+7*8*stride, stride);
    matrixmul_int8_ss(2, 0, 4);
    mstb(2, (uint64_t *)start_addr_C+4*8*stride_post, stride_post);
    matrixmul_int8_ss(3, 0, 5);
    mstb(3, (uint64_t *)start_addr_C+5*8*stride_post, stride_post);
    matrixmul_int8_ss(1, 0, 6);
    mstb(1, (uint64_t *)start_addr_C+6*8*stride_post, stride_post);
    matrixmul_int8_ss(2, 0, 7);
    mstb(2, (uint64_t *)start_addr_C+7*8*stride_post, stride_post);

    mstb(0, (uint64_t *)start_addr_m0+1*8*stride, stride);
}

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

static void matmul_8(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
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
//     matrixmul_int8_ss(0, 2, 3);
//     mstb(0, (uint64_t *)start_addr_C+0*4*8*stride, stride);
//     matrixmul_int8_ss(1, 4, 5);
//     mstb(1, (uint64_t *)start_addr_C+1*4*8*stride, stride);
//     matrixmul_int8_ss(0, 6, 7);
//     mstb(0, (uint64_t *)start_addr_C+2*4*8*stride, stride);
//     mldb_m2((uint64_t *)start_addr_A+3*4*8*stride, stride);
//     mldb_m3((uint64_t *)start_addr_B+3*4*8*stride, stride);
//     mldb_m4((uint64_t *)start_addr_A+4*4*8*stride, stride);
//     mldb_m5((uint64_t *)start_addr_B+4*4*8*stride, stride);
//     mldb_m6((uint64_t *)start_addr_A+5*4*8*stride, stride);
//     mldb_m7((uint64_t *)start_addr_B+5*4*8*stride, stride);
//     matrixmul_int8_ss(1, 2, 3);
//     mstb(1, (uint64_t *)start_addr_C+3*4*8*stride, stride);
//     matrixmul_int8_ss(0, 4, 5);
//     mstb(0, (uint64_t *)start_addr_C+4*4*8*stride, stride);
//     matrixmul_int8_ss(1, 6, 7);
//     mstb(1, (uint64_t *)start_addr_C+5*4*8*stride, stride);

//     mldb_m2((uint64_t *)start_addr_A+0*4*8*stride, stride);
//     mldb_m3((uint64_t *)start_addr_B+0*4*8*stride, stride);
//     mldb_m4((uint64_t *)start_addr_A+1*4*8*stride, stride);
//     mldb_m5((uint64_t *)start_addr_B+1*4*8*stride, stride);
//     matrixmul_int8_ss(0, 2, 3);
//     mstb(0, (uint64_t *)start_addr_C+0*4*8*stride, stride);
//     matrixmul_int8_ss(1, 4, 5);
//     mstb(1, (uint64_t *)start_addr_C+1*4*8*stride, stride);
// }

static void whole_madd(uint64_t *start_addr_A, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //compute 8 times
    uint64_t stride = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(8);
    mcfgni(8);
    mldb_m2((uint64_t *)start_addr_A+0*8*stride, stride);
    mldb_m3((uint64_t *)start_addr_B+0*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_A+1*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+1*8*stride, stride);
    mldb_m6((uint64_t *)start_addr_A+2*8*stride, stride);
    mldb_m7((uint64_t *)start_addr_B+2*8*stride, stride);
    matrixmul_int8_ss(0, 2, 3);
    mstb(0, (uint64_t *)start_addr_C+0*8*stride, stride);
    matrixmul_int8_ss(1, 4, 5);
    mstb(1, (uint64_t *)start_addr_C+1*8*stride, stride);
    matrixmul_int8_ss(0, 6, 7);
    mstb(0, (uint64_t *)start_addr_C+2*8*stride, stride);
    mldb_m2((uint64_t *)start_addr_A+3*8*stride, stride);
    mldb_m3((uint64_t *)start_addr_B+3*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_A+4*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+4*8*stride, stride);
    mldb_m6((uint64_t *)start_addr_A+5*8*stride, stride);
    mldb_m7((uint64_t *)start_addr_B+5*8*stride, stride);
    matrixmul_int8_ss(1, 2, 3);
    mstb(1, (uint64_t *)start_addr_C+3*8*stride, stride);
    matrixmul_int8_ss(0, 4, 5);
    mstb(0, (uint64_t *)start_addr_C+4*8*stride, stride);
    matrixmul_int8_ss(1, 6, 7);
    mstb(1, (uint64_t *)start_addr_C+5*8*stride, stride);

    mldb_m2((uint64_t *)start_addr_A+6*8*stride, stride);
    mldb_m3((uint64_t *)start_addr_B+6*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_A+7*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+7*8*stride, stride);
    matrixmul_int8_ss(0, 2, 3);
    mstb(0, (uint64_t *)start_addr_C+6*8*stride, stride);
    matrixmul_int8_ss(1, 4, 5);
    mstb(1, (uint64_t *)start_addr_C+7*8*stride, stride);
}

int main()
{
    //================================================
    for (uint8_t j = 0; j < 32; j++){
        matmul(token1[j], WQ1, Q1[j]);
    }
    for (uint8_t j = 0; j < 32; j++){
        matmul(token2[j], WQ2, Q2[j]);
    }
    // for (uint8_t j = 0; j < 1; j++){
    //     whole_matmul(token1[j], WQ1, Q1[j]);
    // }

    // for (uint8_t j = 0; j < 1; j++){
    //     whole_matmul(token2[j], WQ2, Q1[j]);
    // }

    // for (uint8_t i = 0; i < 2; i++){
    //     matmul_8(Q1[i], Q2[i], Q3[i]);
    // }
    return 0;
}