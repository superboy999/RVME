/*
 * @Author: superboy
 * @Date: 2024-08-25 23:28:50
 * @LastEditTime: 2024-09-06 00:34:05
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/bert/embeding.c
 * 
 */

#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h" 

#define vocab_size 30522
#define seq_len 256
#define N 256 //indicate the whole matrix elements

#define A_row 32
#define B_col 96



// void initial_matrix()
// {
//     static alignas(32) int8_t A[A_row][N] = { [0 ... A_row-1][0 ... N-1] = 1 };
//     static alignas(32) int8_t B[B_col][N] = { [0 ... B_col-1][0 ... N-1] = 1 };
//     static alignas(32) int8_t C1[32][96][64] { [0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1 };
//     static alignas(32) int8_t C2[32][96][64] { [0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1 };
// }

static alignas(32) int8_t A1[33][N] = { [0 ... 32-1][0 ... N-1] = 1 };
static alignas(32) int8_t B1[96][N] = { [0 ... 96-1][0 ... N-1] = 1 };
static alignas(32) int8_t A2[33][N] = { [0 ... 32-1][0 ... N-1] = 1 };
static alignas(32) int8_t B2[96][N] = { [0 ... 96-1][0 ... N-1] = 1 };
static alignas(32) int8_t C1[33][96][64] = { [0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1 };
static alignas(32) int8_t C2[32][96][64] = { [0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1 };
static alignas(32) int8_t tail_A[2][33][N] = {[0 ... 2-1][0 ... 33-1][0 ... N-1] = 1};
static alignas(32) int8_t tail_B[2][96][N] = {[0 ... 2-1][0 ... 96-1][0 ... N-1] = 1};
static alignas(32) int8_t tail_C[33][96][64] = { [0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1 };
static alignas(32) int8_t tail_C1[33][96][64] = { [0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1 };
static alignas(32) int8_t tail_C2[32][96][64] = { [0 ... 32-1][0 ... 96-1][0 ... 64-1] = 1 };

// static void whole_matmul(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
// {
//     //continuly compute 12
//     uint64_t stride = 32 * sizeof(int8_t); //indicate the row size
//     uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
//     mcfgmi(8);
//     mcfgki(32);
//     mcfgni(8);
//     mldb_m0((uint64_t *)start_addr_m0, stride);
//     mldb_m4((uint64_t *)start_addr_B+0*8*stride, stride);
//     mldb_m5((uint64_t *)start_addr_B+1*8*stride, stride);
//     mldb_m6((uint64_t *)start_addr_B+2*8*stride, stride);
//     mldb_m7((uint64_t *)start_addr_B+3*8*stride, stride);
//     matrixmul_int8_ss(1, 0, 4);
//     mstb(1, (uint64_t *)start_addr_C+0*8*stride_post, stride_post);
//     matrixmul_int8_ss(2, 0, 5);
//     mstb(2, (uint64_t *)start_addr_C+1*8*stride_post, stride_post);
//     matrixmul_int8_ss(3, 0, 6);
//     mstb(3, (uint64_t *)start_addr_C+2*8*stride_post, stride_post);
//     matrixmul_int8_ss(1, 0, 7);
//     mstb(4, (uint64_t *)start_addr_C+3*8*stride_post, stride_post);
//     mldb_m4((uint64_t *)start_addr_B+4*8*stride, stride);
//     mldb_m5((uint64_t *)start_addr_B+5*8*stride, stride);
//     mldb_m6((uint64_t *)start_addr_B+6*8*stride, stride);
//     mldb_m7((uint64_t *)start_addr_B+7*8*stride, stride);
//     matrixmul_int8_ss(2, 0, 4);
//     mstb(2, (uint64_t *)start_addr_C+4*8*stride_post, stride_post);
//     matrixmul_int8_ss(3, 0, 5);
//     mstb(3, (uint64_t *)start_addr_C+5*8*stride_post, stride_post);
//     matrixmul_int8_ss(1, 0, 6);
//     mstb(1, (uint64_t *)start_addr_C+6*8*stride_post, stride_post);
//     matrixmul_int8_ss(2, 0, 7);
//     mstb(2, (uint64_t *)start_addr_C+7*8*stride_post, stride_post);
//     mldb_m4((uint64_t *)start_addr_B+8*8*stride, stride);
//     mldb_m5((uint64_t *)start_addr_B+9*8*stride, stride);
//     mldb_m6((uint64_t *)start_addr_B+10*8*stride, stride);
//     mldb_m7((uint64_t *)start_addr_B+11*8*stride, stride);
//     matrixmul_int8_ss(3, 0, 4);
//     mstb(3, (uint64_t *)start_addr_C+8*8*stride_post, stride_post);
//     matrixmul_int8_ss(1, 0, 5);
//     mstb(1, (uint64_t *)start_addr_C+9*8*stride_post, stride_post);
//     matrixmul_int8_ss(2, 0, 6);
//     mstb(2, (uint64_t *)start_addr_C+10*8*stride_post, stride_post);
//     matrixmul_int8_ss(3, 0, 7);
//     mstb(3, (uint64_t *)start_addr_C+11*8*stride_post, stride_post);
// }

static void whole_matmul(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 12
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
    mldb_m4((uint64_t *)start_addr_B+8*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+8*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+9*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+9*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+10*8*stride, stride);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+10*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+11*8*stride, stride);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+11*8*stride_post, stride_post);
}

static void whole_madd(uint64_t *start_addr_A, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //compute 6 times
    uint64_t stride = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(8);
    mcfgni(8);
    mldb_m4((uint64_t *)start_addr_A+0*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+0*8*stride, stride);
    matrixmul_int8_ss(0, 4, 5);
    mstb(0, (uint64_t *)start_addr_C+0*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_A+1*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+1*8*stride, stride);
    matrixmul_int8_ss(0, 4, 5);
    mstb(0, (uint64_t *)start_addr_C+1*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_A+2*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+2*8*stride, stride);
    matrixmul_int8_ss(0, 4, 5);
    mstb(0, (uint64_t *)start_addr_C+2*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_A+3*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+3*8*stride, stride);
    matrixmul_int8_ss(0, 4, 5);
    mstb(0, (uint64_t *)start_addr_C+3*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_A+4*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+4*8*stride, stride);
    matrixmul_int8_ss(0, 4, 5);
    mstb(0, (uint64_t *)start_addr_C+4*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_A+5*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+5*8*stride, stride);
    matrixmul_int8_ss(0, 4, 5);
    mstb(0, (uint64_t *)start_addr_C+5*8*stride, stride);
}

static void tail_matmul(uint64_t *start_addr_m0, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    //continuly compute 12 times
    uint64_t stride = 26 * sizeof(int8_t); //indicate the elements in one row
    uint64_t stride_1 = 32 * sizeof(int8_t); 
    uint64_t stride_post = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(26);
    mcfgni(8);
    mldb_m0((uint64_t *)start_addr_m0, stride_1);
    mldb_m4((uint64_t *)start_addr_B+0*8*stride_1, stride_1);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+0*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+1*8*stride_1, stride_1);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+1*8*stride_post, stride_post);
    mldb_m6((uint64_t *)start_addr_B+2*8*stride_1, stride_1);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+2*8*stride_post, stride_post);
    mldb_m7((uint64_t *)start_addr_B+3*8*stride_1, stride_1);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+3*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+4*8*stride_1, stride_1);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+4*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+5*8*stride_1, stride_1);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+5*8*stride_post, stride_post);
    mldb_m6((uint64_t *)start_addr_B+6*8*stride_1, stride_1);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+6*8*stride_post, stride_post);
    mldb_m7((uint64_t *)start_addr_B+7*8*stride_1, stride_1);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+7*8*stride_post, stride_post);
    mldb_m4((uint64_t *)start_addr_B+8*8*stride_1, stride_1);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+8*8*stride_post, stride_post);
    mldb_m5((uint64_t *)start_addr_B+9*8*stride_1, stride_1);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+9*8*stride_post, stride_post);
    mldb_m6((uint64_t *)start_addr_B+10*8*stride_1, stride_1);
    matrixmul_int8_ss(1, 0, 4);
    mstb(1, (uint64_t *)start_addr_C+10*8*stride_post, stride_post);
    mldb_m7((uint64_t *)start_addr_B+11*8*stride_1, stride_1);
    matrixmul_int8_ss(1, 0, 5);
    mstb(1, (uint64_t *)start_addr_C+11*8*stride_post, stride_post);
}

static void tail_madd(uint64_t *start_addr_A, uint64_t *start_addr_B, uint64_t *start_addr_C)
{
    uint64_t stride = 8 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(8);
    mcfgni(8);
    mldb_m2((uint64_t *)start_addr_A+0*8*stride, stride);
    matrixmul_int8_ss(0, 2, 3);
    mstb(0, (uint64_t *)start_addr_C+0*8*stride, stride);
    mldb_m3((uint64_t *)start_addr_B+0*8*stride, stride);
    mldb_m4((uint64_t *)start_addr_A+1*8*stride, stride);
    mldb_m5((uint64_t *)start_addr_B+1*8*stride, stride);
    mldb_m6((uint64_t *)start_addr_A+2*8*stride, stride);
    mldb_m7((uint64_t *)start_addr_B+2*8*stride, stride);
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
}

int main()
{
    // printf("embeding test start!");

    //compute for first submatrix
    // for (int8_t i = 0; i < 32; i ++){
    //     for(int8_t j = 0; j < 8; j++){
    //         whole_matmul(A1[i], B1[j], C1[i][j*12]);
    //     }
    // }

    // //compute for second submatrix
    // for (int8_t i = 0; i < 32; i ++){
    //     for(int8_t j = 0; j < 8; j++){
    //         whole_matmul(A2[i], B2[j], C2[i][j*12]);
    //     }
    // }

    // //compute for pointwise
    // for (int8_t i = 0; i < 32; i ++){
    //     for(int8_t j = 0; j <16; j++){
    //         whole_madd(C1[i][6*j], C1[i][6*j], C2[i][j*6]);
    //     }
    // }
    // 以上周期要乘以953/2， 然后还剩下了26

    // // compute for the tail matrix
    // for (int8_t i = 0; i < 8; i ++){
    //     for(int8_t j = 0; j < 8; j++){
    //         tail_matmul(tail_A[0][i], tail_B[0][j*12], tail_C[i][j*12]);
    //     }
    // }

    // //compute for tail pointwise
    // for (int8_t i = 0; i < 32; i ++){
    //     for(int8_t j = 0; j <16; j++){
    //         whole_madd(C1[i][6*j], C1[i][6*j], C2[i][j*6]);
    //     }
    // }

    // pointwise add, plus the position encoding
    for (int8_t i = 0; i < 32; i ++){
        for(int8_t j = 0; j <16; j++){
            whole_madd(C1[i][6*j], tail_C1[i][6*j], tail_C2[i][j*6]);
        }
    }
    // segment encoding
    for (int8_t i = 0; i < 32; i ++){
        for(int8_t j = 0; j <16; j++){
            whole_madd(C1[i][6*j], tail_C1[i][6*j], tail_C2[i][j*6]);
        }
    }
    
    return 0;
}
