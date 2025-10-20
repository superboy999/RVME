/*
 * @Author: superboy
 * @Date: 2025-05-05 21:21:10
 * @LastEditTime: 2025-05-05 23:34:32
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/dataflow/12544x64x147.c
 * 
 */

 #include <stdio.h>
 #include <stdint.h>
 #include <thead_matrix.h>
 #include <stdalign.h>
 #include "../common/inst.h"

#define N 256
static alignas(32) int8_t A[21][1][5][1][8][5][2][N] = {[0 ... 21-1][0 ... 1-1][0 ... 5-1][0 ... 1-1][0 ... 8-1][0 ... 5-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t B[1][1][1][4][5][1][2][N] = {[0 ... 1-1][0 ... 1-1][0 ... 1-1][0 ... 4-1][0 ... 5-1][0 ... 1-1][0 ... 2-1][0 ... N-1] = 1};
static alignas(32) int8_t C[21][1][5][4][8][1][2][N] = {[0 ... 21-1][0 ... 1-1][0 ... 5-1][0 ... 4-1][0 ... 8-1][0 ... 1-1][0 ... 2-1][0 ... N-1] = 0};

int main()
{
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);

    for (uint32_t m = 0; m < 21; m++){
        for (uint32_t a = 0; a < 5; a++){
            for (uint32_t b = 0; b < 4; b++){
                    for (uint32_t x = 0; x < 8; x++){
                        mldb_m4((uint64_t *)A[m][0][a][0][x][0][0], 32);
                        mldb_m5((uint64_t *)A[m][0][a][0][x][0][0], 32);
                        mldb_m6((uint64_t *)A[m][0][a][0][x][0][0], 32);
                        // mldb_m7((uint64_t *)A[m][0][a][0][x][0][0], 32);
                        for (uint32_t z = 0; z < 5; z++){
                            inner_mmul(A[m][0][a][0][x][z][0], B[0][0][0][b][z][0][0]);
                        }
                    }
            }
        }
    }

    return 0;
}