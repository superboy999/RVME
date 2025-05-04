/*
 * @Author: superboy
 * @Date: 2024-08-27 00:18:44
 * @LastEditTime: 2024-09-06 00:50:31
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/bert/attention.c
 * 
 */

#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"

#define N 256 //indicate the whole matrix elements

static alignas(32) int8_t token[2][32][N] = { [0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t WQ[2][8][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t Q1[32][8][64] = {[0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1};
static alignas(32) int8_t Q2[32][8][64] = {[0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1};

static alignas(32) int8_t QQ1[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t K1[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t QKT1[32][32][64] = {[0 ... 32-1][0 ... 32-1][0 ... 64-1] = 1};
static alignas(32) int8_t QKT2[32][32][64] = {[0 ... 32-1][0 ... 32-1][0 ... 64-1] = 1};

static alignas(32) int8_t QK1[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t V[2][8][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t QKV1[32][8][64] = {[0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1};
static alignas(32) int8_t QKV2[32][8][64] = {[0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1};

static alignas(32) int8_t CONC[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t WFC[2][96][N] = {[0 ... 2-1][0 ... 96-1][0 ... N-1] = 1};
static alignas(32) int8_t CONC1[32][96][64] = {[0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1};
static alignas(32) int8_t CONC2[32][96][64] = {[0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1};

static alignas(32) int8_t FF_IN[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t WFF1[2][384][N] = {[0 ... 2-1][0 ... 384-1][0 ... N-1] = 1};
static alignas(32) int8_t WFFB1[32][384][64] = {[0 ... 2-1][0 ... 384-1][0 ... 64-1] = 1};

static alignas(32) int8_t WA[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t WB[2][96][N] = {[0 ... 2-1][0 ... 96-1][0 ... N-1] = 1};
static alignas(32) int8_t WC1[32][96][64] = {[0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1};
static alignas(32) int8_t WC2[32][96][64] = {[0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1};

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
    // printf("attention test");

    // for (uint8_t j = 0; j < 32; j++){
    //     whole_matmul(token[0][j], WQ, Q1[j]);
    // }

    // for (uint8_t j = 0; j < 32; j++){
    //     whole_matmul(token[1][j], WQ, Q2[j]);
    // }

    // for (uint8_t i = 0; i < 32; i++){
    //     whole_madd(Q1[i], Q1[i], Q2[i]);
    // }
//#1
    //以上周期乘以12就算出来Q了，然后再乘以3就得到QKV
    //接下来算Q*K
    
    // for(uint8_t i =0; i < 32; i++){
    //     for(uint8_t j =0; j < 4; j++){
    //         whole_matmul(QQ1[0][i], K1[0][8*j], QKT1[i][8*j]);
    //     }
    // }
    
    // for(uint8_t i =0; i < 32; i++){
    //     for(uint8_t j =0; j < 4; j++){
    //         whole_matmul(QQ1[1][i], K1[1][8*j], QKT2[i][8*j]);
    //     }
    // }

    // for (uint8_t i = 0; i < 32; i++){
    //     whole_madd(QKT1[i], QKT2[i], QKT1[i]);
    // }
//#2
    // 
    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 1; j++){
    //         whole_matmul(QK1[0][i], V[0][8*j], QKV1[i][j]);
    //     }
    // }

    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 1; j++){
    //         whole_matmul(QK1[1][i], V[1][8*j], QKV2[i][j]);
    //     }
    // }

    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 1; j++){
    //         whole_madd(QKV1[i][j], QKV2[i][j], QKV1[i][j]);
    //     }
    // }
//#3
    // 以上周期乘以4，算到QKV
    //以上所有时间乘以12，得到12个头的周期
    // 下面算WFC
    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 12; j++){
    //         whole_matmul(CONC[0][i], WFC[0][8*j], CONC1[i][8*j]);
    //     }
    // }
    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 12; j++){
    //         whole_matmul(CONC[1][i], WFC[1][8*j], CONC2[i][8*j]);
    //     }
    // }

    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 12; j++){
    //         whole_madd(CONC1[i][8*j], CONC2[i][8*j], CONC2[i][8*j]);
    //     }
    // }
//#4
    //以上乘以12是WFC的时间
    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 12; j++){
    //         whole_madd(CONC1[i], CONC2[i][8*j], CONC2[i][8*j]);
    //     }
    // }
//#5
    //以上是第一个残差相加的时间其实是第一个add&norm中的add
    
    //下面计算FF层，但只计算1/2=====================================
    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 48; j++){
            whole_matmul(FF_IN[0][i], WFF1[0][8*j], WFFB1[i][8*j]);
        }
    }
    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 48; j++){
            whole_matmul(FF_IN[1][i], WFF1[1][8*j], WFFB1[i][8*j]);
        }
    }

    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 48; j++){
            whole_madd(WFFB1[1][i], WFFB1[1][8*j], WFFB1[i][8*j]);
        }
    }
//#6
    //以上时间乘以12就是完整的计算Input*W1
    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 12; j++){
    //         whole_matmul(WA[0][i], WB[0][8*j], WC1[i][8*j]);
    //     }
    // } 

    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 12; j++){
    //         whole_matmul(WA[1][i], WB[1][8*j], WC2[i][8*j]);
    //     }
    // } 

    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 12; j++){
    //         whole_madd(WC1[i][8*j], WC2[i][8*j], WC1[i][8*j]);
    //     }
    // } 
//#7
    //以上时间乘以48就是FF的output生成
    //下面是两个偏执的计算
    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 32; j++){
    //         whole_madd(WFFB1[1][8*j], WFFB1[1][8*j], WFFB1[i][8*j]);
    //     }
    // }
    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 12; j++){
    //         whole_madd(CONC1[i], CONC2[i][8*j], CONC2[i][8*j]);
    //     }
    // }
//#8
    //以上时间就是整个FF层的时间(还要加上下面的时间)====================================
    //以下计算第二个add(复用了之前的)
    // for (uint8_t i = 0; i < 32; i++){
    //     for (uint8_t j = 0; j < 12; j++){
    //         whole_madd(CONC1[i], CONC2[i][8*j], CONC1[i][8*j]);
    //     }
    // }
//#9
    //输出层就是嵌入层的计算量
    return 0;

}