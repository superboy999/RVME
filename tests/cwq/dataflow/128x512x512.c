/*
 * @Author: superboy
 * @Date: 2025-04-26 19:47:24
 * @LastEditTime: 2025-04-27 20:07:17
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/dataflow/128x512x512.c
 * 
 */
#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"

#define N 256
static alignas(32) int8_t A[1][1][8][1][1][16][2][N] = {[0 ... 1-1][0 ... 1-1][0 ... 8-1][0 ... 1-1][0 ... 1-1][0 ... 16-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t B[1][4][1][4][16][2][2][N] = {[0 ... 1-1][0 ... 4-1][0 ... 1-1][0 ... 4-1][0 ... 16-1][0 ... 2-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t C[1][4][8][4][1][2][4][N] = {[0 ... 1-1][0 ... 4-1][0 ... 8-1][0 ... 4-1][0 ... 1-1][0 ... 2-1][0 ... 4-1][0 ... N-1] = 0};

int main()
{
   mcfgmi(8);
   mcfgki(32);
   mcfgni(8);

    for (uint32_t n = 0; n < 4; n++){
            for (uint32_t b = 0; b < 4; b++){
                for (uint32_t a = 0; a < 8; a++){
                        for (uint32_t y = 0; y < 2; y++){
                            for (uint32_t z = 0; z < 16; z++){
                                    inner_mmul(A[0][0][a][0][0][z][0], B[0][n][0][b][z][y][0]);
                                }
                        }
                }
            }
    }


   return 0;
}