/*
 * @Author: superboy
 * @Date: 2025-04-09 01:40:26
 * @LastEditTime: 2025-04-09 02:33:35
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/dataflow/256x256x768n.c
 * 
 */
#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"

#define N 256
static alignas(32) int8_t A[1][1][11][24][1][1][3][N] = { [0 ... 1-1][0 ... 1-1][0 ... 11-1][0 ... 24-1][0 ... 1-1][0 ... 1-1][0 ... 3-1][0 ... N-1] = 1};
static alignas(32) int8_t B[1][8][24][1][1][4][1][N] = { [0 ... 1-1][0 ... 8-1][0 ... 24-1][0 ... 1-1][0 ... 1-1][0 ... 4-1][0 ... 1-1][0 ... N-1] = 1};
static alignas(32) int8_t C[1][8][11][1][1][4][3][N] = { [0 ... 1-1][0 ... 8-1][0 ... 11-1][0 ... 1-1][0 ... 1-1][0 ... 4-1][0 ... 3-1][0 ... N-1] = 0};


int main()
{
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);

        for (uint32_t n = 0; n < 8; n++){
                for (uint32_t a = 0; a < 11; a++){
                        for (uint32_t c = 0; c < 24; c++){
                            outer_firstN_init(A[0][0][a][c][0][0][0], B[0][n][c][0][0][0][0], C[0][n][a][0][0][0][0]);
                            for (uint32_t y = 1; y < 4; y++){
                                outer_firstN(A[0][0][a][c][0][0][0], B[0][n][c][0][0][y][0], C[0][n][a][0][0][y][0]);
                            }
                        }
                }
        }

    return 0;
}
