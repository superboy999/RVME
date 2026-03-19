/*
 * @Author: superboy
 * @Date: 2024-08-28 14:40:10
 * @LastEditTime: 2024-09-14 03:40:56
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/mobilenetv2/mobilenetv2_2.c
 * 
 */
#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"
#define N 256

//down layer6
static alignas(32) int8_t A1[2][48][N] = {[0 ... 2-1][0 ... 48-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B1[2][24][N] = {[0 ... 2-1][0 ... 24-1][0 ... N-1] = 1};
static alignas(32) int8_t C1[48][24][64] = {[0 ... 48-1][0 ... 24-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC1[48][24][64] = {[0 ... 48-1][0 ... 24-1][0 ... 64-1] = 1};

static alignas(32) int8_t A2[48][N] = { [0 ... 48-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B2[24][N] = {[0 ... 24-1][0 ... N-1] = 1};
static alignas(32) int8_t C2[48][24][64] = {[0 ... 48-1][0 ... 24-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC2[48][24][64] = {[0 ... 48-1][0 ... 24-1][0 ... 64-1] = 1};

static alignas(32) int8_t A3[12][3][N] = { [0 ... 12-1][0 ... 3-1][0 ... N-1] = 1};
static alignas(32) int8_t B3[12][24][N] = {[0 ... 12-1][0 ... 24-1][0 ... N-1] = 1};
static alignas(32) int8_t C3[3][24][64] = {[0 ... 3-1][0 ... 24-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC3[3][24][64] = {[0 ... 3-1][0 ... 24-1][0 ... 64-1] = 1};

//down layer7
static alignas(32) int8_t A4[3][72][N] = {[0 ... 3-1][0 ... 72-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B4[3][24][N] = {[0 ... 3-1][0 ... 24-1][0 ... N-1] = 1};
static alignas(32) int8_t C4[72][24][64] = {[0 ... 72-1][0 ... 24-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC4[72][24][64] = {[0 ... 72-1][0 ... 24-1][0 ... 64-1] = 1};

static alignas(32) int8_t A5[72][N] = { [0 ... 72-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B5[6][N] = {[0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t C5[72][6][64] = {[0 ... 72-1][0 ... 6-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC5[72][6][64] = {[0 ... 72-1][0 ... 6-1][0 ... 64-1] = 1};

static alignas(32) int8_t A6[18][20][N] = { [0 ... 18-1][0 ... 20-1][0 ... N-1] = 1};
static alignas(32) int8_t B6[18][6][N] = {[0 ... 18-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t C6[20][6][64] = {[0 ... 20-1][0 ... 6-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC6[20][6][64] = {[0 ... 20-1][0 ... 6-1][0 ... 64-1] = 1};

static alignas(32) int8_t A7[5][120][N] = {[0 ... 5-1][0 ... 120-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B7[5][6][N] = {[0 ... 5-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t C7[120][6][64] = {[0 ... 120-1][0 ... 6-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC7[120][6][64] = {[0 ... 120-1][0 ... 6-1][0 ... 64-1] = 1};

static alignas(32) int8_t A8[120][N] = { [0 ... 120-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B8[6][N] = {[0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t C8[120][6][64] = {[0 ... 120-1][0 ... 6-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC8[120][6][64] = {[0 ... 120-1][0 ... 6-1][0 ... 64-1] = 1};

static alignas(32) int8_t A9[30][20][N] = { [0 ... 30-1][0 ... 20-1][0 ... N-1] = 1};
static alignas(32) int8_t B9[30][6][N] = {[0 ... 30-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t C9[20][6][64] = {[0 ... 20-1][0 ... 6-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC9[20][6][64] = {[0 ... 20-1][0 ... 6-1][0 ... 64-1] = 1};

//down layer8
static alignas(32) int8_t A10[5][120][N] = {[0 ... 5-1][0 ... 120-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B10[5][6][N] = {[0 ... 5-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t C10[120][6][64] = {[0 ... 120-1][0 ... 6-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC10[120][6][64] = {[0 ... 120-1][0 ... 6-1][0 ... 64-1] = 1};

static alignas(32) int8_t A11[120][N] = { [0 ... 120-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B11[6][N] = {[0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t C11[120][6][64] = {[0 ... 120-1][0 ... 6-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC11[120][6][64] = {[0 ... 120-1][0 ... 6-1][0 ... 64-1] = 1};

static alignas(32) int8_t A12[30][40][N] = { [0 ... 30-1][0 ... 40-1][0 ... N-1] = 1};
static alignas(32) int8_t B12[30][6][N] = {[0 ... 30-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t C12[40][6][64] = {[0 ... 40-1][0 ... 6-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC12[40][6][64] = {[0 ... 40-1][0 ... 6-1][0 ... 64-1] = 1};

//down layer9
static alignas(32) int8_t A13[10][160][N] = { [0 ... 10-1][0 ... 160-1][0 ... N-1] = 1};
static alignas(32) int8_t B13[10][6][N] = {[0 ... 10-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t C13[160][6][64] = {[0 ... 160-1][0 ... 6-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC13[160][6][64] = {[0 ... 160-1][0 ... 6-1][0 ... 64-1] = 1};

//down layer10
static alignas(32) int8_t A14[30][40][N] = { [0 ... 30-1][0 ... 40-1][0 ... N-1] = 1};
static alignas(32) int8_t B14[30][6][N] = {[0 ... 30-1][0 ... 6-1][0 ... N-1] = 1};
static alignas(32) int8_t C14[40][6][64] = {[0 ... 40-1][0 ... 6-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC14[40][6][64] = {[0 ... 40-1][0 ... 6-1][0 ... 64-1] = 1};

//最后一层不做运算了
#define N 256

static void matmul1(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(27);
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

static void matmulk(uint8_t k,uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgk(k);
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

static void matmul6k(uint8_t k,uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 8
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgk(k);
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
    // for (int8_t k = 0; k < 2; k ++){
    //     for (int8_t i = 0; i < 48; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             matmulk(32,A1[k][i], B1[k][7*j], C1[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t i = 0; i < 48; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         whole_madd(C1[i][7*j], CC1[i][7*j], C1[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 48; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         matmulk(9,A2[i], B2[7*j], C2[i][7*j]);
    //     }
    // }

    // for (int8_t k = 0; k < 12; k ++){
    //     for (int8_t i = 0; i < 8; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             matmulk(32,A3[k][i], B3[k][6*j], C3[i][7*j]);
    //         }
    //     }
    // }

    // for (int8_t k = 0; k < 11; k ++){
    //     for (int8_t i = 0; i < 8; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             whole_madd(C3[i][7*j], CC3[i][7*j], C3[i][7*j]);
    //         }
    //     }
    // }
    //以上乘以三就是layer6的所有
//layer6
    // for (int8_t k = 0; k < 3; k ++){
    //     for (int8_t i = 0; i < 72; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             matmulk(32,A4[k][i], B4[k][7*j], C4[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t i = 0; i < 72; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         whole_madd(C4[i][7*j], CC4[i][7*j], C4[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 72; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         whole_madd(C4[i][7*j], CC4[i][7*j], C4[i][7*j]);
    //     }
    // }


    // for (int8_t i = 0; i < 72; i ++){
    //     for(int16_t j = 0; j < 1; j++){
    //         matmul6k(9,A5[i], B5[j], C5[i][j]);
    //     } 
    // }

    // for (int8_t k = 0; k < 18; k ++){
    //     for (int8_t i = 0; i < 20; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(32,A6[k][i], B6[k][j], C6[i][j]);
    //         }
    //     }
    // }
    // for (int8_t k = 0; k < 17; k ++){
    //     for (int8_t i = 0; i < 20; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(8, C6[i][7*j], CC6[i][7*j], C6[i][7*j]);
    //         }
    //     }
    // }
    


 
    // for (int8_t k = 0; k < 5; k ++){
    //     for (int8_t i = 0; i < 120; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(32,A7[k][i], B7[k][j], C7[i][j]);
    //         }
    //     }
    // }
    // for (int8_t k = 0; k < 4; k ++){
    //     for (int8_t i = 0; i < 120; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(8, C7[i][7*j], CC7[i][7*j], C7[i][7*j]);
    //         }
    //     }
    // }

    // for (int8_t i = 0; i < 120; i ++){
    //     for(int16_t j = 0; j < 1; j++){
    //         matmul6k(9,A8[i], B8[j], C8[i][j]);
    //     } 
    // }

    // for (int8_t k = 0; k < 30; k ++){
    //     for (int8_t i = 0; i < 20; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(32,A9[k][i], B9[k][j], C9[i][j]);
    //         }
    //     }
    // }
    // for (int8_t k = 0; k < 29; k ++){
    //     for (int8_t i = 0; i < 20; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(8, C9[i][7*j], CC9[i][7*j], C9[i][7*j]);
    //         }
    //     }
    // }
    // // 这上下两个小bn一样
    // for (int8_t k = 0; k < 5; k ++){
    //     for (int8_t i = 0; i < 120; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(32,A7[k][i], B7[k][j], C7[i][j]);
    //         }
    //     }
    // }
    // for (int8_t k = 0; k < 4; k ++){
    //     for (int8_t i = 0; i < 120; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(8, C7[i][7*j], CC7[i][7*j], C7[i][7*j]);
    //         }
    //     }
    // }

    // for (int8_t i = 0; i < 120; i ++){
    //     for(int16_t j = 0; j < 1; j++){
    //         matmul6k(9,A8[i], B8[j], C8[i][j]);
    //     } 
    // }

    // for (int8_t k = 0; k < 30; k ++){
    //     for (int8_t i = 0; i < 20; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(32,A9[k][i], B9[k][j], C9[i][j]);
    //         }
    //     }
    // }
    // for (int8_t k = 0; k < 29; k ++){
    //     for (int8_t i = 0; i < 20; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(8, C9[i][7*j], CC9[i][7*j], C9[i][7*j]);
    //         }
    //     }
    // }

//layer7

    // for (int8_t k = 0; k < 5; k ++){
    //     for (int8_t i = 0; i < 120; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(32,A10[k][i], B10[k][j], C10[i][j]);
    //         }
    //     }
    // }
    // for (int8_t k = 0; k < 4; k ++){
    //     for (int8_t i = 0; i < 120; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(8, C10[i][7*j], CC10[i][7*j], C10[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t i = 0; i < 120; i ++){
    //     for(int16_t j = 0; j < 1; j++){
    //         matmul6k(9,A11[i], B11[j], C11[i][j]);
    //     } 
    // }

    // for (int8_t k = 0; k < 30; k ++){
    //     for (int8_t i = 0; i < 40; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(32,A12[k][i], B12[k][j], C12[i][j]);
    //         }
    //     }
    // }
    //  for (int8_t k = 0; k < 29; k ++){
    //     for (int8_t i = 0; i < 40; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(8, C12[i][7*j], CC12[i][7*j], C12[i][7*j]);
    //         }
    //     }
    // }

//layer8
    // for (int8_t k = 0; k < 10; k ++){
    //     for (int16_t i = 0; i < 160; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(32,A13[k][i], B13[k][j], C13[i][j]);
    //         }
    //     }
    // }
    // for (int8_t k = 0; k < 9; k ++){
    //     for (int16_t i = 0; i < 160; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul6k(8, C13[i][7*j], CC13[i][7*j], C13[i][7*j]);
    //         }
    //     }
    // }
//layer9 以上
    return 0;
}