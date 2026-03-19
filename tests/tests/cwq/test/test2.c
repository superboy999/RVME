/*
 * @Author: superboy
 * @Date: 2024-07-09 19:04:13
 * @LastEditTime: 2025-10-18 18:41:07
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/test/test2.c
 * 
 */
#include <stdio.h>
#include <stdint.h>
// #include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"
#define N 256 //indicate the whole matrix length

static alignas(64) int8_t x[N] = {16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static alignas(64) int8_t y[N] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2, 
                3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4,3, 4, 3, 4, 3, 4, 3, 4,3, 4, 3, 4, 3, 4, 3, 4, 
                16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,  
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

static alignas(32) int8_t xt[N] = {16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1,
                16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1,
                16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1,
                16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1,
                16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1,
                16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1,
                16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1,
                16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1};
static alignas(32) int8_t yt[N] = {2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1, 2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1,
                2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1, 2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1,
                2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1, 2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1,
                2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1, 2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1,
                2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1, 2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1,
                2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1, 2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1,
                2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1, 2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1,
                2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1, 2, 3, 16, 1, 1, 1, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1};

static alignas(64) int8_t A[32][8] = {[0 ... 32-1][0 ... 8-1] = 1};
static alignas(64) int8_t AR[32][8] = {[0 ... 32-1][0 ... 8-1] = 1};
static alignas(64) volatile uint32_t z[N] = {[0 ... N-1] = 0};
static alignas(64) volatile uint8_t d[N] = {[0 ... N-1] = 0};
static alignas(64) volatile uint8_t e[N] = {[0 ... N-1] = 0};
static alignas(64) volatile uint8_t f[N] = {[0 ... N-1] = 0};
static inline uint32_t le32_at(const uint8_t* p) {
    return  (uint32_t)p[0]
          | ((uint32_t)p[1] << 8)
          | ((uint32_t)p[2] << 16)
          | ((uint32_t)p[3] << 24);
}

void unpack_le_u32_matrix(const uint8_t* src,
                          size_t rows, size_t cols,
                          size_t row_stride_bytes,
                          uint32_t* dst)
{
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t* row = src + r * row_stride_bytes;
        for (size_t c = 0; c < cols; ++c) {
            const uint8_t* p = row + c * 4;  // 每个元素4字节
            dst[r * cols + c] = le32_at(p);
        }
    }
}

int main()
{
    uint64_t stride = 32 * sizeof(int8_t); //indicate the row size
    uint64_t stride_2 = 8 * sizeof(int8_t); //indicate the row size
    msettilemi(8);
    msettileki(32);
    msettileni(8);

    // static alignas(32) uint8_t z8[N * 4]; // 足够大
    mlate8(0, (uint64_t *)A, stride_2);
    mlbte8(1, (uint64_t *)AR, stride_2);
    mlce32(4, (uint64_t *)z, stride);

    // matrixmul_int8_uu(5, 0, 1);
    // msub_w_mm(4, 4, 4);
    mmacc_w_b(4, 0, 1);
    mscte32(4, (uint8_t *)z, /*按mstb定义给出步长或总字节数*/ 32);
    // msce32(4, (uint8_t *)z, /*按mstb定义给出步长或总字节数*/ 32);

    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    printf("Dump first 256 bytes:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%02X ", z[i]);
    //     if ((i+1) % 32 == 0) puts("");
    // }
    uint32_t result_matrix[8*8];
    // unpack_le_u32_matrix(z, /*rows=*/8, /*cols=*/8, /*row_stride_bytes=*/32, result_matrix);
    
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            printf("%u ", z[r*8 + c]);
        }
        puts("");  
    }
    
    // mmaccu_w_b(4, 0, 1);
    // matrixmul_int8_ss(5, 0, 1);
    // matrixmul_int8_ss(6, 0, 1);
    // matrixmul_int8_ss(7, 0, 1);
    // matrixmul_int8_ss(2, 0, 1);
    // matrixmul_int8_ss(3, 0, 1);
    // mstb(4, (uint32_t *)z, 32);

    // printf("Contents of array z after operations:\n");
    // for (int i = 0; i < 64; i++) {
    //     printf("%u ", z[i]);
    //     if ((i + 1) % 8 == 0) {  // 每8个数换一行，更容易阅读
    //         printf("\n");
    //     }
    // }

    
    // memset(z8, 0, sizeof(z8));

    // msce32(4, (uint8_t *)d, /*按mstb定义给出步长或总字节数*/ 32);

    // asm volatile ("" ::: "memory");
    // asm volatile ("fence rw, rw" ::: "memory");
    // printf("Dump second 256 bytes:\n");
    // // for (int i = 0; i < N; i++) {
    // //     printf("%02X ", z[i]);
    // //     if ((i+1) % 32 == 0) puts("");
    // // }

    // uint32_t result_matrix1[8*8];
    // unpack_le_u32_matrix(d, /*rows=*/8, /*cols=*/8, /*row_stride_bytes=*/32, result_matrix1);

    // for (int r = 0; r < 8; ++r) {
    //     for (int c = 0; c < 8; ++c) {
    //         printf("%u ", result_matrix1[r*8 + c]);
    //     }
    //     puts("");  
    // }

    // // mzero(4);
    // mmaccu_w_b(4, 0, 1);
    // msce32(4, (uint8_t *)e, /*按mstb定义给出步长或总字节数*/ 32);
    // msce32(4, (uint8_t *)f, /*按mstb定义给出步长或总字节数*/ 32);
    // asm volatile ("" ::: "memory");
    // asm volatile ("fence rw, rw" ::: "memory");
    // e[0] = e[0] + 0; 
    // uint32_t result_matrix2[8*8];
    // uint32_t result_matrix3[8*8];
    // unpack_le_u32_matrix(e, /*rows=*/8, /*cols=*/8, /*row_stride_bytes=*/32, result_matrix2);

    // for (int r = 0; r < 8; ++r) {
    //     for (int c = 0; c < 8; ++c) {
    //         printf("%u ", result_matrix2[r*8 + c]);
    //     }
    //     puts("");  
    // }

    // unpack_le_u32_matrix(f, /*rows=*/8, /*cols=*/8, /*row_stride_bytes=*/32, result_matrix3);
    // for (int r = 0; r < 8; ++r) {
    //     for (int c = 0; c < 8; ++c) {
    //         printf("%u ", result_matrix3[r*8 + c]);
    //     }
    //     puts("");  
    // }
    // printf("test finish!!\n");
    return 0;
}
