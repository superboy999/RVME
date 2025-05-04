/*
 * @Author: superboy
 * @Date: 2025-04-07 01:10:43
 * @LastEditTime: 2025-04-07 01:35:26
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/dataflow/256x64x512.c
 * 
 */

 #include <stdio.h>
 #include <stdint.h>
 #include <thead_matrix.h>
 #include <stdalign.h>
 #include "../common/inst.h"

#define N 256
static alignas(32) int8_t A[16][1][1][16][2][N] = {[0 ... 16-1][0 ... 1-1][0 ... 1-1][0 ... 16-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t B[1][4][16][1][2][N] = {[0 ... 1-1][0 ... 4-1][0 ... 16-1][0 ... 1-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t C[16][4][1][1][2][N] = {[0 ... 16-1][0 ... 4-1][0 ... 1-1][0 ... 1-1][0 ... 2-1][0 ... N-1] = 0};

int main()
{
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);

    for (uint32_t m = 0; m < 16; m++){
        for (uint32_t n = 0; n < 4; n++){
            for (uint32_t c = 0; c < 16; c++){
                                        inner_mmul(A[m][0][0][c][0], B[0][n][c][0][0]);
            }
        }
    }

    return 0;
}