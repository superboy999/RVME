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

static alignas(32) int32_t x[N] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static alignas(32) int32_t y[N] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                -1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024, -2048, -4096, -8192, -16384, -32768,
                -1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024, -2048, -4096, -8192, -16384, -32768};

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

    uint32_t result_matrix[8*8];

    // static alignas(32) uint8_t z8[N * 4]; // 足够大
    mlce32(4, (uint64_t *)x, stride);
    mlce32(6, (uint64_t *)y, stride);

    for(int i = 1; i < 3; i++) {
        if(i == 1)
            madd_w_mm(7, 6, 4);
        else if(i == 2)
            madd_w_mv_i(7, 6, 4, 0);
        else if(i == 3)
            msub_w_mm(7, 6, 4);
        else if(i == 4)
            msub_w_mv_i(7, 6, 4, 0);
        else if(i == 5)
            mmul_w_mm(7, 6, 4);
        else if(i == 6)
            mmul_w_mv_i(7, 6, 4, 0);
        else if(i == 7)
            mmulh_w_mm(7, 6, 4);
        else if(i == 8)
            mmulh_w_mv_i(7, 6, 4, 0);
        else if(i == 9)
            mmax_w_mm(7, 6, 4);
        else if(i == 10)
            mmax_w_mv_i(7, 6, 4, 0);
        else if(i == 11)
            mumax_w_mm(7, 6, 4);
        else if(i == 12)
            mumax_w_mv_i(7, 6, 4, 0);
        else if(i == 13)
            mmin_w_mm(7, 6, 4);
        else if(i == 14)
            mmin_w_mv_i(7, 6, 4, 0);
        else if(i == 15)
            mumin_w_mm(7, 6, 4);
        else if(i == 16)
            mumin_w_mv_i(7, 6, 4, 0);
        else if(i == 17)
            msrl_w_mm(7, 6, 4);
        else if(i == 18)
            msrl_w_mv_i(7, 6, 4, 0);
        else if(i == 19)
            msll_w_mm(7, 6, 4);
        else if(i == 20)
            msll_w_mv_i(7, 6, 4, 0);
        else if(i == 21)
            msra_w_mm(7, 6, 4);
        else if(i == 22)
            msra_w_mv_i(7, 6, 4, 0);

        msce32(7, (uint8_t *)z, /*按mstb定义给出步长或总字节数*/ 32);
        asm volatile ("" ::: "memory");
        asm volatile ("fence rw, rw" ::: "memory");
        printf("Dump first 256 bytes:\n");
        // uint32_t result_matrix[8*8];
        unpack_le_u32_matrix(z, /*rows=*/8, /*cols=*/8, /*row_stride_bytes=*/32, result_matrix);
        
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                printf("%d ", result_matrix[r*8 + c]);
            }
            puts("");  
        }
    }


    mmov_mm(7, 6);
    
    msce32(7, (uint8_t *)z, /*按mstb定义给出步长或总字节数*/ 32);
    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    printf("Dump first 256 bytes:\n");
    unpack_le_u32_matrix(z, /*rows=*/8, /*cols=*/8, /*row_stride_bytes=*/32, result_matrix);
    
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            printf("%d ", result_matrix[r*8 + c]);
        }
        puts("");  
    }

    mmov_mm(3, 6);

    msae8(3, (uint8_t *)z, /*按mstb定义给出步长或总字节数*/ 32);
    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    printf("Dump first 256 bytes:\n");
    
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 32; ++c) {
            printf("%d ", z[r*32 + c]);
        }
        puts("");  
    }
    
    mmov_mm(6, 3);

    msce32(6, (uint8_t *)z, /*按mstb定义给出步长或总字节数*/ 32);
    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    printf("Dump first 256 bytes:\n");
    unpack_le_u32_matrix(z, /*rows=*/8, /*cols=*/8, /*row_stride_bytes=*/32, result_matrix);
    
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            printf("%d ", result_matrix[r*8 + c]);
        }
        puts("");  
    }
    

    printf("test finish!!\n");
    return 0;
}
