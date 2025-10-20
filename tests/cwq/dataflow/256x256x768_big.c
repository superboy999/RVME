/*
 * @Author: superboy
 * @Date: 2025-04-03 17:11:42
 * @LastEditTime: 2025-05-13 17:46:28
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/dataflow/256x256x768_big.c
 * 
 */
#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"

#define N 256
static alignas(32) int8_t A[1][1][16][1][1][24][2][N] = { [0 ... 1-1][0 ... 1-1][0 ... 16-1][0 ... 1-1][0 ... 1-1][0 ... 24-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t B[1][1][1][1][24][16][2][N] = { [0 ... 1-1][0 ... 1-1][0 ... 1-1][0 ... 1-1][0 ... 24-1][0 ... 16-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t C[1][1][16][1][1][16][2][N] = { [0 ... 1-1][0 ... 1-1][0 ... 16-1][0 ... 1-1][0 ... 1-1][0 ... 16-1][0 ... 2-1][0 ... N-1] = 0};


int main()
{
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);

    for (uint32_t m = 0; m < 1; m++){
        for (uint32_t n = 0; n < 1; n++){
            for (uint32_t k = 0; k < 1; k++){
                for (uint32_t a = 0; a < 16; a++){
                    for (uint32_t b = 0; b < 1; b++){
                        for (uint32_t c = 0; c < 1; c++){
                            for (uint32_t x = 0; x < 1; x++){
                                for (uint32_t y = 0; y < 16; y++){
                                        mldb_m4((uint64_t *)C[m][n][a][b][x][y][0], 32);
                                        mldb_m5((uint64_t *)C[m][n][a][b][x][y][0], 32);
                                        mldb_m6((uint64_t *)C[m][n][a][b][x][y][0], 32);
                                        mldb_m7((uint64_t *)C[m][n][a][b][x][y][0], 32);
                                    for (uint32_t z = 0; z < 24; z++){

                                        inner_mmul(A[m][k][a][c][x][z][0], B[k][n][c][b][z][y][0]);
                                        // for (uint32_t a = 1; a < 2; a++){
                                        //     inner_mmul(A[m][k][x][z][a], B[k][n][z][y][a]);
                                        // }

                                    }
                                        mstb(4, (uint64_t *)C[m][n][a][b][x][y][0], 32);
                                        mstb(5, (uint64_t *)C[m][n][a][b][x][y][0], 32);
                                        mstb(6, (uint64_t *)C[m][n][a][b][x][y][0], 32);
                                        mstb(7, (uint64_t *)C[m][n][a][b][x][y][0], 32);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}