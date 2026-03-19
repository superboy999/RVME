/*
 * @Author: superboy
 * @Date: 2024-09-08 20:52:12
 * @LastEditTime: 2024-09-08 22:09:54
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/test/128.c
 * 
 */

#include <stdio.h>
#include <stdint.h>
#include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"
#define N 64 //indicate the whole matrix length

int main()
{
    uint64_t stride = 16 * sizeof(int8_t); //indicate the row size
    mcfgmi(4);
    mcfgki(16);
    mcfgni(4);
    static alignas(16) int8_t x[N] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    static alignas(16) int8_t y[N] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    
    static alignas(16) int8_t z[N] = {[0 ... N-1] = 1};
    mldb_m0((uint64_t *)x, stride);
    mldb_m1((uint64_t *)y, stride);

    matrixmul_int8_ss(2, 0, 1);
    matrixmul_int8_ss(3, 0, 1);
    // matrixmul_int8_ss(4, 0, 1);
    // matrixmul_int8_ss(5, 0, 1);
    // matrixmul_int8_ss(6, 0, 1);
    // matrixmul_int8_ss(7, 0, 1);
    // matrixmul_int8_ss(2, 0, 1);
    // matrixmul_int8_ss(3, 0, 1);
    mstb(2, (uint64_t *)z, stride);

    printf("Contents of array x after operations:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", z[i]);
        if ((i+1) % 16 == 0) {  // 每16个数换一行，更容易阅读
            printf("\n");
        }
    }
    
    printf("test finish!!\n");
    return 0;
}