/*
 * @Author: superboy
 * @Date: 2025-04-07 01:52:14
 * @LastEditTime: 2025-04-09 20:00:05
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/dataflow/256x512x2048.c
 * 
 */
#include <stdio.h>
 #include <stdint.h>
 #include <thead_matrix.h>
 #include <stdalign.h>
 #include "../common/inst.h"

#define N 256
static alignas(32) int8_t A[1][3][16][1][1][22][2][N] = {[0 ... 1-1][0 ... 3-1][0 ... 16-1][0 ... 1-1][0 ... 1-1][0 ... 22-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t B[3][32][1][1][22][1][2][N] = {[0 ... 3-1][0 ... 32-1][0 ... 1-1][0 ... 1-1][0 ... 22-1][0 ... 1-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t C[1][32][16][1][1][1][2][N] = {[0 ... 1-1][0 ... 32-1][0 ... 16-1][0 ... 1-1][0 ... 1-1][0 ... 1-1][0 ... 2-1][0 ... N-1] = 0};


int main()
{
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);

    // for (uint32_t k = 0; k < 3; k++){
    //     for (uint32_t n = 0; n < 32; n++){
    //         for (uint32_t m = 0; m < 1; m++){
    //             for (uint32_t a = 0; a < 16; a++){
    //                 for (uint32_t b = 0; b < 1; b++){
    //                     for (uint32_t c = 0; c < 1; c++){
    //                         for (uint32_t y = 0; y < 1; y++){
    //                             for (uint32_t z = 0; z < 22; z++){
    //                                 for (uint32_t x = 0; x < 1; x++){
    //                                     inner_mmul(A[m][k][a][c][x][z][0], B[k][n][c][b][z][y][0]);
    //                                     // for (uint32_t a = 1; a < 2; a++){
    //                                     //     inner_mmul(A[m][k][x][z][a], B[k][n][z][y][a]);
    //                                     // }
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    for (uint32_t k = 0; k < 3; k++){
        for (uint32_t n = 0; n < 32; n++){
                for (uint32_t a = 0; a < 16; a++){
                                for (uint32_t z = 0; z < 22; z++){
                                        inner_mmul(A[0][0][0][0][0][0][0], B[0][0][0][0][0][0][0]);

                                }
                }
        }
    }

    return 0;
}