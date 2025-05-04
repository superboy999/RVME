/*
 * @Author: superboy
 * @Date: 2025-04-09 22:07:50
 * @LastEditTime: 2025-04-27 23:55:09
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/dataflow/128x2048x512.c
 * 
 */

 #include <stdio.h>
 #include <stdint.h>
 #include <thead_matrix.h>
 #include <stdalign.h>
 #include "../common/inst.h"

#define N 256
// static alignas(32) int8_t A[1][1][16][16][1][16][1][N] = {[0 ... 1-1][0 ... 1-1][0 ... 16-1][0 ... 16-1][0 ... 1-1][0 ... 16-1][0 ... 1-1][0 ... N-1] = 1};
// static alignas(32) int8_t B[1][8][1][8][16][1][3][N] = {[0 ... 1-1][0 ... 8-1][0 ... 1-1][0 ... 8-1][0 ... 16-1][0 ... 1-1][0 ... 3-1][0 ... N-1] = 1};
// static alignas(32) int8_t C[1][8][16][8][1][1][3][N] = {[0 ... 1-1][0 ... 8-1][0 ... 16-1][0 ... 8-1][0 ... 1-1][0 ... 1-1][0 ... 3-1][0 ... N-1] = 0};
static alignas(32) int8_t A[1][1][8][1][1][16][2][N] = {[0 ... 1-1][0 ... 1-1][0 ... 8-1][0 ... 1-1][0 ... 1-1][0 ... 16-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t B[1][11][1][12][16][1][2][N] = {[0 ... 1-1][0 ... 11-1][0 ... 1-1][0 ... 12-1][0 ... 16-1][0 ... 1-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t C[1][11][8][12][1][1][4][N] = {[0 ... 1-1][0 ... 11-1][0 ... 8-1][0 ... 12-1][0 ... 1-1][0 ... 1-1][0 ... 4-1][0 ... N-1] = 0};

int main()
{
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);

        // for (uint32_t n = 0; n < 8; n++){
        //         for (uint32_t a = 0; a < 11; a++){
        //                 for (uint32_t c = 0; c < 24; c++){
        //                     outer_firstN_init(A[0][0][a][c][0][0][0], B[0][n][c][0][0][0][0], C[0][n][a][0][0][0][0]);
        //                     for (uint32_t y = 1; y < 4; y++){
        //                         outer_firstN(A[0][0][a][c][0][0][0], B[0][n][c][0][0][y][0], C[0][n][a][0][0][y][0]);
        //                     }
        //                 }
        //         }
        // }

        for (uint32_t n = 0; n < 11; n++){
                for (uint32_t a = 0; a < 8; a++){
                    for (uint32_t b = 0; b < 12; b++){
                                for (uint32_t z = 0; z < 16; z++){
                                        inner_mmul(A[0][0][a][0][0][z][0], B[0][n][0][b][z][0][0]);
                                        // for (uint32_t a = 1; a < 2; a++){
                                        //     inner_mmul(A[m][k][x][z][a], B[k][n][z][y][a]);
                                        // }
                                    }
                    }
                }
    }

    return 0;
}