/*
 * @Author: superboy
 * @Date: 2025-04-05 12:10:22
 * @LastEditTime: 2025-04-06 22:01:22
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/dataflow/256x256x768.c
 * 
 */

 #include <stdio.h>
 #include <stdint.h>
 #include <thead_matrix.h>
 #include <stdalign.h>
 #include "../common/inst.h"

#define N 256
static alignas(32) int8_t A[16][1][1][1][1][24][2][N] = { [0 ... 16-1][0 ... 1-1][0 ... 1-1][0 ... 1-1][0 ... 1-1][0 ... 24-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t B[1][1][1][16][24][1][2][N] = { [0 ... 1-1][0 ... 1-1][0 ... 1-1][0 ... 16-1][0 ... 24-1][0 ... 1-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t C[16][1][1][16][1][1][2][N] = { [0 ... 16-1][0 ... 1-1][0 ... 1-1][0 ... 16-1][0 ... 1-1][0 ... 1-1][0 ... 2-1][0 ... N-1] = 0};


int main()
{
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);

    // for (uint32_t m = 0; m < 16; m++){
    //     for (uint32_t n = 0; n < 1; n++){
    //         for (uint32_t k = 0; k < 1; k++){
    //             for (uint32_t a = 0; a < 1; a++){
    //                 for (uint32_t b = 0; b < 16; b++){
    //                     for (uint32_t c = 0; c < 1; c++){
    //                         for (uint32_t y = 0; y < 1; y++){
    //                             for (uint32_t z = 0; z < 24; z++){
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

    for (uint32_t m = 0; m < 16; m++){
                    for (uint32_t b = 0; b < 16; b++){
                                for (uint32_t z = 0; z < 24; z++){
                                        inner_mmul(A[m][0][0][0][0][z][0], B[0][0][0][b][z][0][0]);
                                        // for (uint32_t a = 1; a < 2; a++){
                                        //     inner_mmul(A[m][k][x][z][a], B[k][n][z][y][a]);
                                        // }
                                    }
                    }
    }

    return 0;
}
