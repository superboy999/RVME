/*
 * @Author: superboy
 * @Date: 2024-09-03 21:29:43
 * @LastEditTime: 2024-09-03 22:42:49
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/transformer_test/tiling_test.c
 * 
 */

#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"

#define N 256 //indicate the whole matrix elements

static alignas(32) int8_t token[1000][N] = {[0 ... 1000-1][0 ... N-1] = 1};
static alignas(32) int8_t vocab1[1000][N] = {[0 ... 1000-1][0 ... N-1] = 1};
static alignas(32) int8_t vocab2[1000][N] = {[0 ... 1000-1][0 ... N-1] = 1};
static alignas(32) int8_t result[64] = {[0 ... 64-1] = 1};
static void matmul(uint64_t *start_addr_A, uint64_t *start_addr_B)
{   
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size
    mcfgmi(8);
    mcfgki(32);
    mcfgni(8);
    mldb_m0((uint64_t *)start_addr_A, stride);
    mldb_m1((uint64_t *)start_addr_B, stride);
    matrixmul_int8_ss(2, 0, 1);
    mcfgki(8);
    matrixmul_int8_ss(0, 2, 1);
}

int main()
{
    for(uint32_t i = 0; i < 1000; i++){
        matmul(token[i], vocab1[i]);
    }
    // mstb(4, result, 8);
    for(uint32_t i = 0; i < 1000; i++){
        matmul(token[i], vocab2[i]);
    }
    // mstb(4, result, 8);
    return 0;
}

