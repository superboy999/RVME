/*
 * @Author: superboy
 * @Date: 2025-10-18 13:42:39
 * @LastEditTime: 2025-10-18 13:52:03
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/test/test1.c
 * 
 */
/*
 * @Author: superboy
 * @Date: 2024-07-09 19:04:13
 * @LastEditTime: 2025-10-18 04:21:40
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/tests/cwq/test/test.c
 * 
 */
#include <stdio.h>
#include <stdint.h>
// #include <thead_matrix.h>
#include <stdalign.h>
#include "../common/inst.h"
#define N 256 //indicate the whole matrix length

static alignas(32) int8_t x[N] = {16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static alignas(32) int8_t y[N] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2, 
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

static alignas(32) volatile uint8_t z[N] = {[0 ... N-1] = 0};
static alignas(32) volatile uint8_t d[N] = {[0 ... N-1] = 0};


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
    uint64_t stridex = 8 * sizeof(int8_t); //indicate the row size
    msettilemi(8);
    msettileki(32);
    msettileni(8);
    // static alignas(32) uint8_t z8[N * 4]; // 足够大
    dmaload_spm((uint64_t *)x, 0, 0);
    dmaload_spm((uint64_t *)y, 1024, 0);
    mlae8_spm(0, 0, stride, 0);
    mlbe8_spm(1, 1024, stride, 0);

    // matrixmul_int8_uu(5, 0, 1);
    mmaccu_w_b(4, 0, 1);
    mmaccu_w_b(5, 0, 1);
    msce32(4, (uint8_t *)z, /*按mstb定义给出步长或总字节数*/ 32);

    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    printf("Dump first 256 bytes:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%02X ", z[i]);
    //     if ((i+1) % 32 == 0) puts("");
    // }
    uint32_t result_matrix[8*8];
    unpack_le_u32_matrix(z, /*rows=*/8, /*cols=*/8, /*row_stride_bytes=*/32, result_matrix);
    
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            printf("%u ", result_matrix[r*8 + c]);
        }
        puts("");  
    }
    
    madd_w_mm(6, 4, 5);

    msce32(6, (uint8_t *)d, /*按mstb定义给出步长或总字节数*/ 32);

    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    printf("Dump second 256 bytes:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%02X ", z[i]);
    //     if ((i+1) % 32 == 0) puts("");
    // }

    uint32_t result_matrix1[8*8];
    unpack_le_u32_matrix(d, /*rows=*/8, /*cols=*/8, /*row_stride_bytes=*/32, result_matrix1);

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            printf("%u ", result_matrix1[r*8 + c]);
        }
        puts("");  
    }
    msub_w_mm(6, 4, 5);
    msce32(6, (uint8_t *)d, /*按mstb定义给出步长或总字节数*/ 32);
    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    printf("Dump second 256 bytes:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%02X ", z[i]);
    //     if ((i+1) % 32 == 0) puts("");
    // }

    unpack_le_u32_matrix(d, /*rows=*/8, /*cols=*/8, /*row_stride_bytes=*/32, result_matrix1);

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            printf("%u ", result_matrix1[r*8 + c]);
        }
        puts("");  
    }

    mmul_w_mm(6, 4, 5);
    msce32(6, (uint8_t *)d, /*按mstb定义给出步长或总字节数*/ 32);
    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    printf("Dump second 256 bytes:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%02X ", z[i]);
    //     if ((i+1) % 32 == 0) puts("");
    // }

    unpack_le_u32_matrix(d, /*rows=*/8, /*cols=*/8, /*row_stride_bytes=*/32, result_matrix1);

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            printf("%u ", result_matrix1[r*8 + c]);
        }
        puts("");  
    }

    msra_w_mm(6, 4, 5);
    msce32(6, (uint8_t *)d, /*按mstb定义给出步长或总字节数*/ 32);
    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    printf("Dump second 256 bytes:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%02X ", z[i]);
    //     if ((i+1) % 32 == 0) puts("");
    // }
    unpack_le_u32_matrix(d, /*rows=*/8, /*cols=*/8, /*row_stride_bytes=*/32, result_matrix1);

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            printf("%u ", result_matrix1[r*8 + c]);
        }
        puts("");  
    }
    printf("test finish!!\n");
    return 0;
}
