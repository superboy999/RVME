#ifndef __MATMUL_H__
#define __MATMUL_H__

#include "inst.h"
#include "config_v2_3.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

// int countnum = 0;

static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
    const elem_t* A, const elem_t* B, const acc_t* D, acc_t* C,
    size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C) {

    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    enum tiled_matmul_type_t tiled_matmul_type = OS;
    size_t tile_I = 16;
    size_t tile_J = 16;
    size_t tile_K = 32;

    msettilemi(DIM_I);
    msettileni(DIM_J);
    msettileki(DIM_K);

    mlce32_spm(TILE_NUM+0, 262144, 512, 0);
    mlce32_spm(TILE_NUM+1, 262144, 512, 0);
    mlce32_spm(TILE_NUM+2, 262144, 512, 0);
    mlce32_spm(TILE_NUM+3, 262144, 512, 0);
        
    for (size_t j = 0; j < 16; j+= 8){
        for (size_t k0 = 0; k0 < 1024; k0++){
            // printf("[DEBUG] load B: k0 = %d, j = %d\n", k0, j);
            const elem_t* B_dram_addr = B + k0 * 128 + j * 8;
            uint64_t B_sp_addr = 131072 + (uint64_t)(128 * k0 + 8 * j);
            // printf("B_sp_addr = %d\n", B_sp_addr);
            dmaload_spm((uint64_t)B_dram_addr, B_sp_addr,1);
        }
    }

    for (size_t i = 0; i < 16; i++){
        for (size_t k = 0; k < 32; k++){
            // printf("[DEBUG] load A: i = %d, k = %d\n", i, k);
            const elem_t* A_dram_addr = A + i * 8192 + k * 256;
            uint64_t A_sp_addr = (uint64_t)(8192 * i + 256 * k);
            // printf("A_sp_addr = %d\n", A_sp_addr);
            dmaload_spm((uint64_t)A_dram_addr, A_sp_addr,4);
        }
    }
    msync_spm();
    for (size_t i = 0; i < 16; i+=2) {
        for (size_t j = 0; j < 16; j+=2) {
            // printf("[DEBUG] compute: i = %d, j = %d\n", i, j);
            uint64_t C00_sp_addr = 4096 * i + 32 * j + 262144;
            uint64_t C01_sp_addr = 4096 * i + 32 * j + 262176;
            uint64_t C10_sp_addr = 4096 * i + 32 * j + 266240;
            uint64_t C11_sp_addr = 4096 * i + 32 * j + 266272;
            if(i != 0 || j != 0) {
                // mlce32_spm(TILE_NUM+0, C00_sp_addr, 512, 0);
                mzero(TILE_NUM+0);
                // printf("C00_sp_addr = %d\n", C00_sp_addr);
                // printf("C01_sp_addr = %d\n", C01_sp_addr);
                // mlce32_spm(TILE_NUM+1, C01_sp_addr, 512, 0);
                mzero(TILE_NUM+1); /* acc1 = 0 */
                // printf("C10_sp_addr = %d\n", C10_sp_addr);
                // mlce32_spm(TILE_NUM+2, C10_sp_addr, 512, 0);
                mzero(TILE_NUM+2); /* acc2 = 0 */
                // printf("C11_sp_addr = %d\n", C11_sp_addr);
                // mlce32_spm(TILE_NUM+3, C11_sp_addr, 512, 0);
                mzero(TILE_NUM+3); /* acc3 = 0 */
            }
            for (size_t k = 0; k < 32; k++) {
                // printf("[DEBUG] mlae/be, i = %d, j = %d, k = %d\n", i, j, k);
                mlae8_spm(0, 8192*i+k*256, 32, 0);
                // mmov_mm(2, 0);
                // mlae8_spm(2, 8192*i+k*256+8192, 32, 0);
                mlae8_spm(1, 8192*i+k*256+8192, 32, 0);
                mlbe8_spm(2, 4096*k+8*j+131072, 128, 0);
                // mmov_mm(1, 2);
                // mmov_mm(3, 1);
                // mlae8_spm(3, 8192*i+k*256+8192, 32, 0);
                mlbe8_spm(3, 4096*k+8*j+131080, 128, 0);
                // printf("[DEBUG] macc, i = %d, j = %d, k = %d\n", i, j, k);
                mmacc_w_b(4, 2, 0);
                mmacc_w_b(6, 2, 1);
                mmacc_w_b(5, 3, 0);
                mmacc_w_b(7, 3, 1);
            //     if(k % 8 == 0) {
            //         for(int i = 0; i < 32; i++) {
            //         countnum += (i + 2) * (i + 6);
            //         countnum += (j + i) * (i + 2);
            //         countnum = countnum / 2;
            //     }                
            // }
            }
            msce32_spm(4, C00_sp_addr, 512, 0);
            msce32_spm(5, C01_sp_addr, 512, 0);
            msce32_spm(6, C10_sp_addr, 512, 0);
            msce32_spm(7, C11_sp_addr, 512, 0);
        }
    }
    msync_spm();
    // printf("[DEBUG] dma store C\n");
    for(size_t i0 = 0; i0 < 128; i0++){
        for(size_t j = 0; j < 16; j+= 2){
            const acc_t* C_dram_addr = C + i0 * 128 + j * 8;
            uint64_t C_sp_addr = 262144 + (uint64_t)(512 * i0 + 32 * j);
            dmastore_spm((uint64_t)C_dram_addr, C_sp_addr, 1);
        }
    }

    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
}


#endif

