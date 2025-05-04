/*
 * @Author: superboy
 * @Date: 2025-03-13 17:22:01
 * @LastEditTime: 2025-04-03 20:02:08
 * @LastEditors: superboy
 * @Description: test tile size = [256, 768, 3072]
 * @FilePath: /gem5-rvm/tests/cwq/dataflow/test.c
 * 
 */

#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"

#define N 256
static alignas(32) int8_t A[2][8][1][2][16][6][N] = { [0 ... 2-1][0 ... 8-1][0 ... 1-1][0 ... 2-1][0 ... 16-1][0 ... 6-1][0 ... N-1] = 1}; // 3215392-4001824
static alignas(32) int8_t B[8][2][2][24][6][2][N] = { [0 ... 8-1][0 ... 1-1][0 ... 2-1][0 ... 24-1][0 ... 6-1][0 ... 2-1][0 ... N-1] = 1}; // 856096-3215392
static alignas(32) int8_t C[2][2][1][24][16][2][N] = {[0 ... 2-1][0 ... 2-1][0 ... 1-1][0 ... 24-1][0 ... 16-1][0 ... 2-1][0 ... N-1] = 1}; //69664-856096

int main()
{
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);
    for (uint64_t i = 0; i < 2; i++){
        for (uint64_t j = 0; j < 1; j++){
            for (uint64_t k = 0; k < 2; k++){
                for (uint64_t z = 0; z < 2; z++){
                    for (uint64_t x = 0; x < 24; x++){
                        for (uint64_t y = 0; y < 1; y++){

                            for (uint64_t a = 0; a < 5; a++){
                                for (uint64_t s = 0; s < 6; s++){
                                    outer_mmul3_init(A[k][j][y][z][a*3][s], B[j][i][z][x][s][0], C[k][i][y][x][a*3][0],0, 0);
                                    for (uint64_t d = 0; d < 2; d++){
                                        // printf("1\n");
                                        outer_mmul3(A[k][j][y][z][a*3][s], B[j][i][z][x][s][d], C[k][i][y][x][a*3][d],0);
                                    }
                                }
                            }

                            // for (uint64_t a = 5; a < 6; a++){
                            //     for (uint64_t s = 0; s < 6; s++){
                            //         for (uint64_t d = 0; d < 2; d++){
                            //             // printf("1\n");
                            //             outer_mmul1(A[k][j][y][z][a*3][s], B[j][i][z][x][s][d], C[k][i][y][x][a*3][d], 0);
                            //         }
                            //     }
                            // }

                        }
                    }
                }
            }
        }
    }


    // for (uint64_t i = 0; i < 10; i++){


    // for (uint64_t a = 0; a < 5; a++){
    //     for (uint64_t s = 0; s < 6; s++){
    //         outer_mmul3_init(A[0][0][0][0][a*3][s], B[0][0][0][0][s][0], C[0][0][0][0][a*3][0], 1536, 0);
    //         // mldb_m4((uint64_t *)B[0][0][0][0][s][0], 32);
    //         // matrixmul_int8_ss(5, 0, 4);
    //         // mstb(5,(uint64_t *)C[0][0][0][0][a][0], 32);
    //         // matrixmul_int8_ss(6, 1, 4);
    //         // mstb(6,(uint64_t *)C[0][0][0][0][a][0], 32);
    //         // mldb_m1((uint64_t *)A[0][0][0][0][a][s], 32);
    //         // matrixmul_int8_ss(5, 0, 4);
    //         // matrixmul_int8_ss(6, 1, 4);

    //         outer_mmul3(A[0][0][0][0][a*3][s], B[0][0][0][0][s][0], C[0][0][0][0][a*3][0], 0);
    //         outer_mmul3(A[0][0][0][0][a*3][s], B[0][0][0][0][s][0], C[0][0][0][0][a*3][0], 0);
    //         outer_mmul3(A[0][0][0][0][a*3][s], B[0][0][0][0][s][0], C[0][0][0][0][a*3][0], 0);
    //         outer_mmul3(A[0][0][0][0][a*3][s], B[0][0][0][0][s][0], C[0][0][0][0][a*3][0], 0);
    //         outer_mmul3(A[0][0][0][0][a*3][s], B[0][0][0][0][s][0], C[0][0][0][0][a*3][0], 0);
    //         outer_mmul3(A[0][0][0][0][a*3][s], B[0][0][0][0][s][0], C[0][0][0][0][a*3][0], 0);
    //         outer_mmul3(A[0][0][0][0][a*3][s], B[0][0][0][0][s][0], C[0][0][0][0][a*3][0], 0);
    //         // for (uint64_t d = 0; d < 1; d++){
    //         //     printf("1\n");
    //         //     outer_mmul3(A[0][0][0][0][a][s], B[0][0][0][0][s][d], C[0][0][0][0][a][d], 0);
    //         // }
    //     }
    // }
    // }
    return 0;
}