/*
 * @Author: superboy
 * @Date: 2024-08-27 23:37:02
 * @LastEditTime: 2024-09-14 03:35:15
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/mobilenetv2/mobilenetv2.c
 * 
 */

//=============
//
//=============
#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"

#define N 256

static alignas(32) int8_t A1[1][1568][N] = { [0 ... 1-1][0 ... 1568-1][0 ... N-1] = 1};
static alignas(32) int8_t B1[1][4][N] = {[0 ... 1-1][0 ... 4-1][0 ... N-1] = 1};
static alignas(32) int8_t C1[1568][4][64] = {[0 ... 1568-1][0 ... 4-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC1[1568][4][64] = {[0 ... 1568-1][0 ... 4-1][0 ... 64-1] = 1};

static alignas(32) int8_t A2[1568][N] = { [0 ... 1568-1][0 ... N-1] = 1};
static alignas(32) int8_t B2[4][N] = {[0 ... 4-1][0 ... N-1] = 1};
static alignas(32) int8_t C2[1568][4][64] = {[0 ... 1568-1][0 ... 4-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC2[1568][4][64] = {[0 ... 1568-1][0 ... 4-1][0 ... 64-1] = 1};

static alignas(32) int8_t A3[1568][N] = { [0 ... 1568-1][0 ... N-1] = 1};
static alignas(32) int8_t B3[4][N] = {[0 ... 4-1][0 ... N-1] = 1};
static alignas(32) int8_t C3[1568][4][64] = {[0 ... 1568-1][0 ... 4-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC3[1568][4][64] = {[0 ... 1568-1][0 ... 4-1][0 ... 64-1] = 1};

static alignas(32) int8_t A4[1568][N] = { [0 ... 1568-1][0 ... N-1] = 1};
static alignas(32) int8_t B4[12][N] = {[0 ... 12-1][0 ... N-1] = 1};
static alignas(32) int8_t C4[1568][12][64] = {[0 ... 1568-1][0 ... 12-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC4[1568][12][64] = {[0 ... 1568-1][0 ... 12-1][0 ... 64-1] = 1};

static alignas(32) int8_t A5[392][N] = { [0 ... 392-1][0 ... N-1] = 1};
static alignas(32) int8_t B5[12][N] = {[0 ... 12-1][0 ... N-1] = 1};
static alignas(32) int8_t C5[392][12][64] = {[0 ... 392-1][0 ... 12-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC5[392][12][64] = {[0 ... 392-1][0 ... 12-1][0 ... 64-1] = 1};

static alignas(32) int8_t A6[3][392][N] = { [0 ... 3-1][0 ... 392-1][0 ... N-1] = 1};
static alignas(32) int8_t B6[3][3][N] = {[0 ... 3-1][0 ... 3-1][0 ... N-1] = 1};
static alignas(32) int8_t C6[392][3][64] = {[0 ... 392-1][0 ... 3-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC6[392][3][64] = {[0 ... 392-1][0 ... 3-1][0 ... 64-1] = 1};

static alignas(32) int8_t A7[392][N] = { [0 ... 392-1][0 ... N-1] = 1};
static alignas(32) int8_t B7[18][N] = {[0 ... 18-1][0 ... N-1] = 1};
static alignas(32) int8_t C7[392][18][64] = {[0 ... 392-1][0 ... 18-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC7[392][18][64] = {[0 ... 392-1][0 ... 18-1][0 ... 64-1] = 1};

static alignas(32) int8_t A8[4][392][N] = { [0 ... 4-1][0 ... 3-1][0 ... N-1] = 1}; //原本是4.5
static alignas(32) int8_t B8[4][3][N] = {[0 ... 4-1][0 ... 3-1][0 ... N-1] = 1};
static alignas(32) int8_t C8[392][3][64] = {[0 ... 3-1][0 ... 3-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC8[392][3][64] = {[0 ... 3-1][0 ... 3-1][0 ... 64-1] = 1};
// down layer4
static alignas(32) int8_t A9[18][N] = {[0 ... 18-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B9[392][N] = {[0 ... 392-1][0 ... N-1] = 1};
static alignas(32) int8_t C9[18][392][64] = {[0 ... 18-1][0 ... 392-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC9[18][392][64] = {[0 ... 18-1][0 ... 392-1][0 ... 64-1] = 1};

static alignas(32) int8_t A10[18][N] = { [0 ... 18-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B10[98][N] = {[0 ... 98-1][0 ... N-1] = 1};
static alignas(32) int8_t C10[18][98][64] = {[0 ... 18-1][0 ... 98-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC10[18][98][64] = {[0 ... 18-1][0 ... 98-1][0 ... 64-1] = 1};

static alignas(32) int8_t A11[4][4][N] = { [0 ... 4-1][0 ... 4-1][0 ... N-1] = 1};//原本是4.5
static alignas(32) int8_t B11[4][98][N] = {[0 ... 4-1][0 ... 98-1][0 ... N-1] = 1};
static alignas(32) int8_t C11[4][98][64] = {[0 ... 3-1][0 ... 98-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC11[4][98][64] = {[0 ... 3-1][0 ... 98-1][0 ... 64-1] = 1};

static alignas(32) int8_t A12[24][N] = {[0 ... 24-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B12[98][N] = {[0 ... 98-1][0 ... N-1] = 1};
static alignas(32) int8_t C12[24][98][64] = {[0 ... 24-1][0 ... 98-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC12[24][98][64] = {[0 ... 24-1][0 ... 98-1][0 ... 64-1] = 1};

static alignas(32) int8_t A13[24][N] = { [0 ... 24-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B13[98][N] = {[0 ... 98-1][0 ... N-1] = 1};
static alignas(32) int8_t C13[24][98][64] = {[0 ... 24-1][0 ... 98-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC13[24][98][64] = {[0 ... 24-1][0 ... 98-1][0 ... 64-1] = 1};

static alignas(32) int8_t A14[6][4][N] = { [0 ... 6-1][0 ... 4-1][0 ... N-1] = 1};
static alignas(32) int8_t B14[6][98][N] = {[0 ... 6-1][0 ... 98-1][0 ... N-1] = 1};
static alignas(32) int8_t C14[4][98][64] = {[0 ... 3-1][0 ... 98-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC14[4][98][64] = {[0 ... 3-1][0 ... 98-1][0 ... 64-1] = 1};

//down layer5
static alignas(32) int8_t A15[24][N] = {[0 ... 24-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B15[98][N] = {[0 ... 98-1][0 ... N-1] = 1};
static alignas(32) int8_t C15[24][98][64] = {[0 ... 24-1][0 ... 98-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC15[24][98][64] = {[0 ... 24-1][0 ... 98-1][0 ... 64-1] = 1};

static alignas(32) int8_t A16[24][N] = { [0 ... 24-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B16[24][N] = {[0 ... 24-1][0 ... N-1] = 1};
static alignas(32) int8_t C16[24][24][64] = {[0 ... 24-1][0 ... 24-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC16[24][24][64] = {[0 ... 24-1][0 ... 24-1][0 ... 64-1] = 1};

static alignas(32) int8_t A17[6][8][N] = { [0 ... 6-1][0 ... 8-1][0 ... N-1] = 1};//原本是4.5
static alignas(32) int8_t B17[6][24][N] = {[0 ... 4-1][0 ... 24-1][0 ... N-1] = 1};
static alignas(32) int8_t C17[8][24][64] = {[0 ... 8-1][0 ... 24-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC17[8][24][64] = {[0 ... 8-1][0 ... 24-1][0 ... 64-1] = 1};

static alignas(32) int8_t A18[2][48][N] = {[0 ... 2-1][0 ... 48-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B18[2][24][N] = {[0 ... 2-1][0 ... 24-1][0 ... N-1] = 1};
static alignas(32) int8_t C18[48][24][64] = {[0 ... 48-1][0 ... 24-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC18[48][24][64] = {[0 ... 48-1][0 ... 24-1][0 ... 64-1] = 1};

static alignas(32) int8_t A19[48][N] = { [0 ... 48-1][0 ... N-1] = 1}; 
static alignas(32) int8_t B19[24][N] = {[0 ... 24-1][0 ... N-1] = 1};
static alignas(32) int8_t C19[48][24][64] = {[0 ... 48-1][0 ... 24-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC19[48][24][64] = {[0 ... 48-1][0 ... 24-1][0 ... 64-1] = 1};

static alignas(32) int8_t A20[12][8][N] = { [0 ... 12-1][0 ... 8-1][0 ... N-1] = 1};
static alignas(32) int8_t B20[12][24][N] = {[0 ... 6-1][0 ... 24-1][0 ... N-1] = 1};
static alignas(32) int8_t C20[8][24][64] = {[0 ... 3-1][0 ... 24-1][0 ... 64-1] = 1};
static alignas(32) int8_t CC20[8][24][64] = {[0 ... 3-1][0 ... 24-1][0 ... 64-1] = 1};

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

static void matmul4k(uint8_t k,uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
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
}

static void matmul2k(uint8_t k,uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
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
}

static void matmul3k(uint8_t k,uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
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
    // for (int16_t i = 0; i < 1568; i ++){
    //     for(int16_t j = 0; j < 1; j++){
    //         matmul4k(27, A1[0][i], B1[0][j*8], C1[i][8*j]);
    //     }
    // }

    // delete here
    // for (int8_t i = 0; i < 4; i ++){
    //     for(int16_t j = 0; j < 196; j++){
    //         whole_madd(C1[i][j*7], CC1[i][j*7], C1[i][7*j]);
    //     }
    // }
    // delete here

//#1 layer1
    // for (int16_t i = 0; i < 1568; i ++){
    //     for(int8_t j = 0; j < 1; j++){
    //         matmul4k(32,A2[i], B2[j], C2[i][j]);
    //     }
    // }

    // for (int8_t i = 0; i < 4; i ++){
    //     for(int16_t j = 0; j < 196; j++){
    //         whole_madd(C2[i][7*j], CC2[i][7*j], C2[i][7*j]);
    //     }
    // }

    // for (int16_t i = 0; i < 1568; i ++){
    //     for(int8_t j = 0; j < 1; j++){
    //         matmul4k(9, A3[i], B3[j], C3[i][j]);
    //     }
    // }
    // for (int8_t i = 0; i < 4; i ++){
    //     for(int16_t j = 0; j < 196; j++){
    //         whole_madd(C3[i][7*j], CC3[i][7*j], C3[i][7*j]);
    //     }
    // }
    // for (int16_t i = 0; i < 1568; i ++){
    //     for(int16_t j = 0; j < 1; j++){
    //         matmul2k(32, A3[i], B3[j], C3[i][j]);
    //     }
    // }
    // for (int8_t i = 0; i < 2; i ++){
    //     for(int16_t j = 0; j < 196; j++){
    //         whole_madd(C3[i][7*j], CC3[i][7*j], C3[i][7*j]);
    //     }
    // }
//#2  layer2
    // for (int16_t i = 0; i < 1568; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         matmul4k(16,A4[i], B4[4*j], C4[i][4*j]);
    //     }
    // }
    // for (int16_t i = 0; i < 392; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         matmul4k(9,A5[i], B5[4*j], C5[i][4*j]);
    //     }
    // }
    // for (int16_t k = 0; k < 3; k ++){
    //     for (int16_t i = 0; i < 392; i ++){
    //         for(int16_t j = 0; j < 1; j++){
    //             matmul3k(32,A6[k][i], B6[k][j], C6[i][j]);
    //         }
    //     }
    // }
    // for (int16_t i = 0; i < 392; i ++){
    //     for(int16_t j = 0; j < 1; j++){
    //         matmul3k(8, C6[i][7*j], CC6[i][7*j], C6[i][7*j]);
    //     }
    // }
    // for (int16_t i = 0; i < 392; i ++){
    //     for(int16_t j = 0; j < 1; j++){
    //         matmul3k(8,C6[i][7*j], CC6[i][7*j], C6[i][7*j]);
    //     }
    // }
//#3 

    // for (int16_t i = 0; i < 392; i ++){
    //     for(int8_t j = 0; j < 6; j++){
    //         matmul3k(24,A7[i], B7[3*j], C7[i][3*j]);
    //     }
    // }
    // for (int16_t i = 0; i < 392; i ++){
    //     for(int8_t j = 0; j < 6; j++){
    //         matmul3k(9,A7[i], B7[3*j], C7[i][3*j]);
    //     }
    // }
    for (int16_t k = 0; k < 4; k ++){
        for (int16_t i = 0; i < 392; i ++){
            for(int16_t j = 0; j < 1; j++){
                matmul3k(32,A8[k][i], B8[k][7*j], C8[i][7*j]);
            }
        }
    }
    for (int16_t i = 0; i < 392; i ++){
        for(int16_t j = 0; j < 1; j++){
            matmul3k(8,C8[i][7*j], CC8[i][7*j], C8[i][7*j]);
        }
    }
    for (int16_t i = 0; i < 392; i ++){
        for(int16_t j = 0; j < 1; j++){
            matmul3k(8,C8[i][7*j], CC8[i][7*j], C8[i][7*j]);
        }
    }
    for (int16_t i = 0; i < 392; i ++){
        for(int16_t j = 0; j < 1; j++){
            matmul3k(8,C8[i][7*j], CC8[i][7*j], C8[i][7*j]);
        }
    }
    // return 0;
//#4 layer3
    // for (int8_t i = 0; i < 18; i ++){
    //     for(int16_t j = 0; j < 49; j++){
    //         matmulk(24,A9[i], B9[7*j], C9[i][7*j]);
    //     }
    // }

    // for (int8_t i = 0; i < 18; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         matmulk(9,A10[i], B10[7*j], C10[i][7*j]);
    //     }
    // }


    // for (int8_t k = 0; k < 4; k ++){
    //     for (int8_t i = 0; i < 4; i ++){
    //         for(int16_t j = 0; j < 11; j++){
    //             matmulk(32,A11[0][i], B11[0][8*j], C11[i][8*j]);
    //         }
    //     }
    // }
    // for (int8_t i = 0; i < 4; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         whole_madd(C11[i][7*j], CC11[i][7*j], C11[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 4; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         whole_madd(C11[i][7*j], CC11[i][7*j], C11[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 4; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         whole_madd(C11[i][7*j], CC11[i][7*j], C11[i][7*j]);
    //     }
    // }


    // for (int8_t i = 0; i < 24; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         matmulk(32,A12[i], B12[6*j], C12[i][6*j]);
    //     }
    // }

    // for (int8_t i = 0; i < 24; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         matmulk(9,A13[0], B13[7*j], C13[i][7*j]);
    //     }
    // }

    // for (int8_t k = 0; k < 6; k ++){
    //     for (int8_t i = 0; i < 4; i ++){
    //         for(int16_t j = 0; j < 12; j++){
    //             matmulk(32,A14[k][i], B14[k][7*j], C14[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t i = 0; i < 4; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         whole_madd(C14[i][7*j], CC14[i][7*j], C14[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 4; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         whole_madd(C14[i][7*j], CC14[i][7*j], C14[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 4; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         whole_madd(C14[i][7*j], CC14[i][7*j], C14[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 4; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         whole_madd(C14[i][7*j], CC14[i][7*j], C14[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 4; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         whole_madd(C14[i][7*j], CC14[i][7*j], C14[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 4; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         whole_madd(C14[i][7*j], CC14[i][7*j], C14[i][7*j]);
    //     }
    // }



    // for (int8_t i = 0; i < 24; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         matmulk(32,A12[i], B12[6*j], C12[i][6*j]);
    //     }
    // }


    // for (int8_t i = 0; i < 24; i ++){
    //     for(int16_t j = 0; j < 12; j++){
    //         matmulk(9,A13[i], B13[7*j], C13[i][7*j]);
    //     }
    // }

    // for (int8_t k = 0; k < 6; k ++){
    //     for (int8_t i = 0; i < 4; i ++){
    //         for(int16_t j = 0; j < 12; j++){
    //             matmulk(32,A14[k][i], B14[k][7*j], C14[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t i = 0; i < 6; i ++){
    //     for(int16_t j = 0; j < 4; j++){
    //         whole_madd(C14[i][7*j], CC14[i][7*j], C14[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 6; i ++){
    //     for(int16_t j = 0; j < 4; j++){
    //         whole_madd(C14[i][7*j], CC14[i][7*j], C14[i][7*j]);
    //     }
    // }

    // for (int8_t i = 0; i < 6; i ++){
    //     for(int16_t j = 0; j < 4; j++){
    //         whole_madd(C14[i][7*j], CC14[i][7*j], C14[i][7*j]);
    //     }
    // }

    // for (int8_t i = 0; i < 6; i ++){
    //     for(int16_t j = 0; j < 4; j++){
    //         whole_madd(C14[i][7*j], CC14[i][7*j], C14[i][7*j]);
    //     }
    // }

    // for (int8_t i = 0; i < 6; i ++){
    //     for(int16_t j = 0; j < 4; j++){
    //         whole_madd(C14[i][7*j], CC14[i][7*j], C14[i][7*j]);
    //     }
    // }

// layer4
    // for (int8_t i = 0; i < 24; i ++){
    //     for(int8_t j = 0; j < 12; j++){
    //         matmulk(32,A15[i], B15[7*j], C15[i][7*j]);
    //     }
    // }

    // for (int8_t i = 0; i < 24; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         matmulk(9,A16[i], B16[7*j], C16[i][7*j]);
    //     }
    // }

    // for (int8_t k = 0; k < 6; k ++){
    //     for (int8_t i = 0; i < 8; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             matmulk(32,A17[k][i], B17[k][7*j], C17[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t k = 0; k < 5; k ++){
    //     for (int8_t i = 0; i < 8; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             whole_madd(C17[i][7*j], CC17[i][7*j], C17[i][7*j]);
    //         }
    //     }
    // }

    // for (int8_t k = 0; k < 2; k ++){
    //     for (int8_t i = 0; i < 48; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             matmulk(32,A18[k][i], B18[k][7*j], C18[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t i = 0; i < 48; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         whole_madd(C18[i][7*j], CC18[i][7*j], C18[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 48; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         matmulk(9,A19[i], B19[7*j], C19[i][7*j]);
    //     }
    // }

    // for (int8_t k = 0; k < 12; k ++){
    //     for (int8_t i = 0; i < 8; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             matmulk(32,A20[k][i], B20[k][7*j], C20[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t k = 0; k < 11; k ++){
    //     for (int8_t i = 0; i < 8; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             whole_madd(C20[i][7*j], CC20[i][7*j], C20[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t k = 0; k < 2; k ++){
    //     for (int8_t i = 0; i < 48; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             matmulk(32,A18[k][i], B18[k][7*j], C18[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t i = 0; i < 48; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         whole_madd(C18[i][7*j], CC18[i][7*j], C18[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 48; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         matmulk(9,A19[i], B19[7*j], C19[i][7*j]);
    //     }
    // }
    // for (int8_t k = 0; k < 12; k ++){
    //     for (int8_t i = 0; i < 8; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             matmulk(32,A20[k][i], B20[k][7*j], C20[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t k = 0; k < 11; k ++){
    //     for (int8_t i = 0; i < 8; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             whole_madd(C20[i][7*j], CC20[i][7*j], C20[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t k = 0; k < 2; k ++){
    //     for (int8_t i = 0; i < 48; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             matmulk(32,A18[k][i], B18[k][7*j], C18[i][7*j]);
    //         }
    //     }
    // }
    // for (int8_t i = 0; i < 48; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         whole_madd(C18[i][7*j], CC18[i][7*j], C18[i][7*j]);
    //     }
    // }
    // for (int8_t i = 0; i < 48; i ++){
    //     for(int16_t j = 0; j < 3; j++){
    //         matmulk(9,A19[i], B19[7*j], C19[i][7*j]);
    //     }
    // }

    // for (int8_t k = 0; k < 12; k ++){
    //     for (int8_t i = 0; i < 8; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             matmulk(32,A20[k][i], B20[k][7*j], C20[i][7*j]);
    //         }
    //     }
    // }

    // for (int8_t k = 0; k < 11; k ++){
    //     for (int8_t i = 0; i < 8; i ++){
    //         for(int16_t j = 0; j < 3; j++){
    //             whole_madd(C20[i][7*j], CC20[i][7*j], C20[i][7*j]);
    //         }
    //     }
    // }
//#layer5
    
    return 0;

}