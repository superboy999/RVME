/*
 * @Author: superboy
 * @Date: 2024-08-18 23:46:06
 * @LastEditTime: 2024-08-18 23:46:07
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/transformer/real_encoder.c
 * 
 */
//==========================
// input size: 256*512
// head:8
// 6 layers
//==========================
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

static alignas(32) int8_t QQ1[2][8][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t K1[2][8][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t QKT1[32][32][64] = {[0 ... 32-1][0 ... 32-1][0 ... 64-1] = 1};
static alignas(32) int8_t QKT2[32][32][64] = {[0 ... 32-1][0 ... 32-1][0 ... 64-1] = 1};

static alignas(32) int8_t QK1[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t V[2][8][N] = {[0 ... 2-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t QKV1[32][8][64] = {[0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1};
static alignas(32) int8_t QKV2[32][8][64] = {[0 ... 32-1][0 ... 8-1][0 ... 64-1] = 1};

static alignas(32) int8_t CONC[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t WFC[2][64][N] = {[0 ... 2-1][0 ... 64-1][0 ... N-1] = 1};
static alignas(32) int8_t CONC1[32][64][64] = {[0 ... 32-1][0 ... 64-1][0 ... 64-1] = 1};
static alignas(32) int8_t CONC2[32][64][64] = {[0 ... 32-1][0 ... 64-1][0 ... 64-1] = 1};

static alignas(32) int8_t FF_IN[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t WFF1[2][256][N] = {[0 ... 2-1][0 ... 256-1][0 ... N-1] = 1};
static alignas(32) int8_t WFFB1[32][256][64] = {[0 ... 2-1][0 ... 256-1][0 ... 64-1] = 1};

static alignas(32) int8_t WA[2][32][N] = {[0 ... 2-1][0 ... 32-1][0 ... N-1] = 1};
static alignas(32) int8_t WB[2][64][N] = {[0 ... 2-1][0 ... 64-1][0 ... N-1] = 1};
static alignas(32) int8_t WC1[32][64][64] = {[0 ... 32-1][0 ... 64-1][0 ... 64-1] = 1};
static alignas(32) int8_t WC2[32][64][64] = {[0 ... 32-1][0 ... 64-1][0 ... 64-1] = 1};

static void whole_matmul(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);
    mldb_m0((uint64_t *)start_addr_m0, stride);
    mldb_m4((uint64_t *)start_addr_B+0*4*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+1*4*8*stride, stride);
    mldb_m6((uint64_t *)start_addr_B+2*4*8*stride, stride);
    mldb_m7((uint64_t *)start_addr_B+3*4*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*4*8*stride, stride);
    matrixmul_int8_ss(2, 0, 5);
    mstb(2, (uint64_t *)start_addr_C+1*4*8*stride, stride);
    matrixmul_int8_ss(3, 0, 6);
    mstb(3, (uint64_t *)start_addr_C+2*4*8*stride, stride);
    matrixmul_int8_ss(1, 0, 7);
    mstb(4, (uint64_t *)start_addr_C+3*4*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_B+4*4*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+5*4*8*stride, stride);
    mldb_m6((uint64_t *)start_addr_B+6*4*8*stride, stride);
    mldb_m7((uint64_t *)start_addr_B+7*4*8*stride, stride);
    matrixmul_int8_ss(2, 0, 4);
    mstb(2, (uint64_t *)start_addr_C+4*4*8*stride, stride);
    matrixmul_int8_ss(3, 0, 5);
    mstb(3, (uint64_t *)start_addr_C+5*4*8*stride, stride);
    matrixmul_int8_ss(1, 0, 6);
    mstb(1, (uint64_t *)start_addr_C+6*4*8*stride, stride);
    matrixmul_int8_ss(2, 0, 7);
    mstb(2, (uint64_t *)start_addr_C+7*4*8*stride, stride);

    mstb(0, (uint64_t *)start_addr_m0+1*4*8*stride, stride);
}
static void whole_madd(uint64_t *start_addr_A, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //compute 8 times
    uint64_t stride = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(8);
    mcfgni(8);
    mldb_m2((uint64_t *)start_addr_A+0*4*8*stride, stride);
    mldb_m3((uint64_t *)start_addr_B+0*4*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_A+1*4*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+1*4*8*stride, stride);
    mldb_m6((uint64_t *)start_addr_A+2*4*8*stride, stride);
    mldb_m7((uint64_t *)start_addr_B+2*4*8*stride, stride);
    madd(0, 2, 3);
    mstb(0, (uint64_t *)start_addr_C+0*4*8*stride, stride);
    madd(1, 4, 5);
    mstb(1, (uint64_t *)start_addr_C+1*4*8*stride, stride);
    madd(0, 6, 7);
    mstb(0, (uint64_t *)start_addr_C+2*4*8*stride, stride);
    mldb_m2((uint64_t *)start_addr_A+3*4*8*stride, stride);
    mldb_m3((uint64_t *)start_addr_B+3*4*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_A+4*4*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+4*4*8*stride, stride);
    mldb_m6((uint64_t *)start_addr_A+5*4*8*stride, stride);
    mldb_m7((uint64_t *)start_addr_B+5*4*8*stride, stride);
    madd(1, 2, 3);
    mstb(1, (uint64_t *)start_addr_C+3*4*8*stride, stride);
    madd(0, 4, 5);
    mstb(0, (uint64_t *)start_addr_C+4*4*8*stride, stride);
    madd(1, 6, 7);
    mstb(1, (uint64_t *)start_addr_C+5*4*8*stride, stride);

    mldb_m2((uint64_t *)start_addr_A+0*4*8*stride, stride);
    mldb_m3((uint64_t *)start_addr_B+0*4*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_A+1*4*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+1*4*8*stride, stride);
    madd(0, 2, 3);
    mstb(0, (uint64_t *)start_addr_C+0*4*8*stride, stride);
    madd(1, 4, 5);
    mstb(1, (uint64_t *)start_addr_C+1*4*8*stride, stride);
}

int main()
{
    //================================================
    for (uint8_t j = 0; j < 32; j++){
        whole_matmul(token[0][j], WQ[j], Q1[j]);
    }

    for (uint8_t j = 0; j < 32; j++){
        whole_matmul(token[1][j], WQ[j], Q2[j]);
    }

    for (uint8_t i = 0; i < 32; i++){
        whole_madd(Q1[i], Q1[i], Q2[i]);
    }
    // 以上周期乘以8就算出来Q了，然后乘以3就得到QKV
    //接下来算Q*KT
    for(uint8_t j =0; j < 8; j++){
        whole_matmul(QQ1[0][j], K1[0][j], QKT1[j]);
    }
    
    for(uint8_t j =0; j < 8; j++){
        whole_matmul(QQ1[1][j], K1[1][j], QKT2[j]);
    }

    for (uint8_t i = 0; i < 32; i++){
        whole_madd(QKT1[i], QKT2[i], QKT1[i]);
    }
    // 以上周期乘以4，算到QKT
    // 
    for (uint8_t j = 0; j < 32; j++){
        whole_matmul(QK1[0][j], V[0][j], QKV1[j]);
    }

    for (uint8_t j = 0; j < 32; j++){
        whole_matmul(QK1[1][j], V[1][j], QKV2[j]);
    }

    for (uint8_t i = 0; i < 32; i++){
        whole_madd(QKV1[i], QKV2[i], QKV1[i]);
    }

    // 以上周期乘以4，算到QKV
    //以上所有时间乘以8，得到8个头的周期======================
    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 8; j++){
            whole_matmul(CONC[0][i], WFC[0][8*j], CONC1[i][8*j]);
        }
    }
    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 8; j++){
            whole_matmul(CONC[1][i], WFC[1][8*j], CONC2[i][8*j]);
        }
    }

    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 8; j++){
            whole_matmul(CONC1[i][8*j], CONC2[i][8*j], CONC1[i][8*j]);
        }
    }
    // 以上乘以8就是concat的周期
    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 8; j++){
            whole_matmul(CONC1[i][8*j], CONC2[i][8*j], CONC1[i][8*j]);
        }
    }
    //以上是第一个add&norm中的add
    
    //下面计算FF层=====================================
    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 32; j++){
            whole_matmul(FF_IN[0][i], WFF1[0][8*j], WFFB1[i][8*j]);
        }
    }
    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 32; j++){
            whole_matmul(FF_IN[1][i], WFF1[1][8*j], WFFB1[i][8*j]);
        }
    }

    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 32; j++){
            whole_madd(WFFB1[i][j*8], WFFB1[i][8*j], WFFB1[i][8*j]);
        }
    }
    //以上时间乘以8就是完整的计算Input*W1
    //下面是加了第一个偏执
    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 32; j++){
            whole_madd(WFFB1[i][8*j], WFFB1[i][8*j], WFFB1[i][8*j]);
        }
    }
    //再计算output========
    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 8; j++){
            whole_matmul(WA[0][i], WB[0][8*j], WC1[i][8*j]);
        }
    } 

    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 8; j++){
            whole_matmul(WA[1][i], WB[1][8*j], WC2[i][8*j]);
        }
    } 

    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 8; j++){
            whole_madd(WC1[i][8*j], WC2[i][8*j], WC1[i][8*j]);
        }
    } 
    //以上时间乘以32就是FF的output生成========
    //下面再加第二个偏执
    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 8; j++){
            whole_madd(WC1[i][8*j], WC2[i][8*j], WC1[i][8*j]);
        }
    } 
    //以上时间就是整个FF层的时间====================================

    //以下计算第二个add&norm的add
    for (uint8_t i = 0; i < 32; i++){
        for (uint8_t j = 0; j < 8; j++){
            whole_madd(WC1[i][8*j], WC2[i][8*j], WC1[i][8*j]);
        }
    } 
    return 0;
}