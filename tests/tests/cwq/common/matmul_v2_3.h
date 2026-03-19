/* 完整 double-buffer OS matmul 实现
 * 结构保持：
 *   tiled_matmul_auto  : 决定 tile 尺寸（double buffer 容量模型）
 *   tiled_matmul       : 参数校验与调度
 *   tiled_matmul_outer : 外层 tile_I/J/K 循环
 *   sp_tiled_matmul_os : 内层计算 + double buffer DMA/compute
 */

#ifndef __MATMUL_H__
#define __MATMUL_H__

#include "inst.h"
#include "config_v2_3.h"
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

/* =============================
 *  Inner kernel: OS + double buffer
 * ============================= */
static void sp_tiled_matmul_os(
    const elem_t *A, const elem_t *B,
    const acc_t  *D, acc_t *C,
    size_t I, size_t J, size_t K,
    size_t stride_A, size_t stride_B,
    size_t stride_D, size_t stride_C,
    bool no_bias, 
    int current_buffer, int next_buffer,
    int c_current_buffer, int c_next_buffer,
    size_t i0, size_t j0, size_t k0,
    size_t I0, size_t J0, size_t K0,
    const elem_t *A1, const elem_t *B1)
{
    // printf("[INFO][MATMUL] Start 2x2 OS sp_tiled_matmul_os_double_buffer_pipeline(): i0 = %d, j0 = %d, k0 = %d, I=%zu, J=%zu, K=%zu, buffer_state=%d/%d, c_buffer_state = %d/%d, C_dramAddr = %d\n", 
    //        i0, j0, k0, I, J, K, current_buffer, next_buffer, c_current_buffer, c_next_buffer, C);
    int A_blocks = (int)8;
    int B_blocks = (int)(MAX_DMA_BLOCK / (DIM_J * sizeof(elem_t)));
    int D_blocks = (int)(MAX_DMA_BLOCK / (DIM_J * sizeof(acc_t)));
    int C_blocks = (int)(MAX_DMA_BLOCK / (DIM_J * sizeof(acc_t)));
    if (A_blocks < 1) A_blocks = 1;
    if (B_blocks < 1) B_blocks = 1;
    if (D_blocks < 1) D_blocks = 1;
    if (C_blocks < 1) C_blocks = 1;
    if (A_blocks > (int)K) A_blocks = (int)K;
    if (B_blocks > (int)J) B_blocks = (int)J;
    if (D_blocks > (int)J) D_blocks = (int)J;
    if (C_blocks > (int)J) C_blocks = (int)J;
    // printf("[INFO][MATMUL] A_blocks = %d, B_blocks = %d, D_blocks = %d, C_blocks = %d\n", A_blocks, B_blocks, D_blocks, C_blocks);

    /* 第1阶段：加载下一套ABD 到 next_buffer */
    /* 不是全局的最后一个tile块*/
    // if(k0 == (K0-1) && !(i0 == (I0-1) && j0 == (J0-1))){
      // /* ---------- bias ---------- */
      // if (D != NULL && !no_bias) {
      //   printf("[DEBUG][MATMUL] Phase 0: Loading bias to C buffer %d\n", current_buffer);
      //   for (size_t j = 0; j < J; j += (size_t)D_blocks) {
      //     for (size_t i0 = 0; i0 < I * DIM_I; i0++) {
      //       printf("[DEBUG] Loading next D for i0 = %d, j = %d\n", i0, j);
      //       size_t cols = min_size(J - j, (size_t)D_blocks);
      //       const elem_t *D_dram = D + j * DIM_J;  
      //       uint64_t D_sp = get_C_block_addr(next_buffer, i0 , j, J);
      //       size_t bytes = cols * DIM_J * sizeof(elem_t);
      //       uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
      //       if (cachelines >= 32) cachelines = 0;
      //       dmaload_spm((uint64_t)D_dram, D_sp, 1);
      //     }
      //   }
      // }
      /* ---------- 给C的本部分buffer清零 ---------- */
      if (k0 == 0) {
        // printf("[DEBUG][MATMUL] Phase 0: Cleaning C_currentbuffer %d\n", c_current_buffer);
        for (size_t j = 0; j < J; j += (size_t)D_blocks) {
          for (size_t i0 = 0; i0 < I * DIM_I; i0++) {
            // printf("[DEBUG] Loading current D for i0 = %d, j = %d\n", i0, j);
            size_t cols = min_size(J - j, (size_t)D_blocks);
            const elem_t *D_dram = D + j * DIM_J;  
            uint64_t D_sp = get_C_block_addr(c_current_buffer, i0 , j, J);
            size_t bytes = cols * DIM_J * sizeof(elem_t);
            uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
            if (cachelines >= 32) cachelines = 0;
            dmaload_spm((uint64_t)D_dram, D_sp, 1);
          }
        }
      }

    
    // printf("[DEBUG][MATMUL] Phase 1: Loading next A and B data to buffer %d\n", next_buffer);

    if(!(i0 == I0 && j0 == J0 && k0 == K0)){
      /* 加载下一个tile块A的数据 */
      for (size_t i = 0; i < I; i++) {
        for (size_t k = 0; k < K; k += 1) {
          // printf("[DEBUG] Loading next A for i = %d, k = %d, K = %d\n", i, k, K);
          // printf("[DEBUG] A = %d, A1 = %d\n", A, A1);
          const elem_t *A_dram = A1 + i * DIM_I * stride_A + k * DIM_K * DIM_I;  
          uint64_t A_sp = get_A_dim_block_addr(next_buffer, i, k, K);
          uint8_t cachelines = (uint8_t)div_up(DIM_I * DIM_K * sizeof(elem_t), CACHELINE_SIZE);
          if (cachelines >= 32) cachelines = 0;
          dmaload_spm((uint64_t)A_dram, A_sp, 4); // 搬入一个A的DIM块        
        }
      }
      
      /* 加载下一个tile块B的数据 */
      for (size_t j = 0; j < J; j += (size_t)B_blocks) {
        for (size_t k0_in = 0; k0_in < K * DIM_K; k0_in++) {
          // printf("[DEBUG] Loading next B for k0 = %d, j = %d\n", k0_in, j);
          size_t cols = min_size(J - j, (size_t)B_blocks);
          const elem_t *B_dram = B1 + j * DIM_J + k0_in * stride_B;  
          uint64_t B_sp = get_B_block_addr(next_buffer, k0_in , j, J);
          size_t bytes = cols * DIM_J * sizeof(elem_t);
          uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
          if (cachelines >= 32) cachelines = 0;
          dmaload_spm((uint64_t)B_dram, B_sp, 1);
        }
      }
    }

    msync_spm();

    /* ---------- compute ---------- */

    // printf("[DEBUG][MATMUL] Phase 2: Compute\n");
    for (size_t i = 0; i < I; i += 2) {
        bool has_i1 = ((i + 1) < I);
        for (size_t j = 0; j < J; j += 2) {
            bool has_j1 = ((j + 1) < J);

            uint64_t c00_sp_addr = get_C_block_addr(c_current_buffer, (i * DIM_I), j, J);
            uint64_t c01_sp_addr = has_j1 ? get_C_block_addr(c_current_buffer, (i* DIM_I), j+1, J) : 0;
            uint64_t c10_sp_addr = has_i1 ? get_C_block_addr(c_current_buffer, (i + 1)*DIM_I, j, J) : 0;
            uint64_t c11_sp_addr = (has_i1 && has_j1) ? get_C_block_addr(c_current_buffer, (i + 1)* DIM_I, j+1, J) : 0;

            /* 1. 初始化累加器：加载bias或清零 */
            if (!no_bias) {
              // printf("[DEBUG][MATMUL] Phase 2.1: Loading bias to C register, i = %d, j = %d\n", i, j);
              mlce32_spm(TILE_NUM+0, c00_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
              if (has_j1) mlce32_spm(TILE_NUM+1, c01_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
              if (has_i1) mlce32_spm(TILE_NUM+2, c10_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
              if (has_i1 && has_j1) mlce32_spm(TILE_NUM+3, c11_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
            } else {
              /* 无bias，清零累加器 */
              // printf("[DEBUG][MATMUL] Phase 2.1.2: Dont have bias: Clearing C register\n");
              mlce32_spm(TILE_NUM+0, c00_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
              msub_w_mm(TILE_NUM+0, TILE_NUM+0, TILE_NUM+0); /* acc0 = 0 */
              
              if (has_j1) {
                  mlce32_spm(TILE_NUM+1, c01_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
                  msub_w_mm(TILE_NUM+1, TILE_NUM+1, TILE_NUM+1); /* acc1 = 0 */
              }
              
              if (has_i1) {
                  mlce32_spm(TILE_NUM+2, c10_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
                  msub_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+2); /* acc2 = 0 */
              }
              
              if (has_i1 && has_j1) {
                  mlce32_spm(TILE_NUM+3, c11_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
                  msub_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+3); /* acc3 = 0 */
              }
            }
              
            /* 2. K循环 - 累加A×B到当前C tile */
            for (size_t k = 0; k < K; k++) {

                // printf("[DEBUG][MATMUL] Phase 2.3: MMACC, i = %d, j = %d, k = %d\n", i, j, k);
                // printf("[DEBUG][MATMUL] Phase 2.3.1: mlae8_spm A00: i = %d, k = %d, K = %d.\n", i, k, K);
                mlae8_spm(0, get_A_dim_block_addr(current_buffer, i, k, K), (int)(DIM_K * sizeof(elem_t)), 0);
                if (has_i1){
                  // printf("[DEBUG][MATMUL] Phase 2.3.1: mlae8_spm A10: i+1 = %d, k = %d, K = %d.\n", i+1, k, K);
                  mlae8_spm(1, get_A_dim_block_addr(current_buffer, i+1, k, K), (int)(DIM_K * sizeof(elem_t)), 0);
                }

                // printf("[DEBUG][MATMUL] Phase 2.3.2: mlbe8_spm B00: k = %d, j = %d, J = %d\n", k, j, J);
                mlbe8_spm(2, get_B_block_addr(current_buffer, k * DIM_K, j, J), (int)(J * DIM_J * sizeof(elem_t)), 0);
                if (has_j1){
                  // printf("[DEBUG][MATMUL] Phase 2.3.2: mlbe8_spm B01: k = %d, j+1 = %d, J = %d\n", k, j+1, J);
                  mlbe8_spm(3, get_B_block_addr(current_buffer, k * DIM_K, j+1, J), (int)(J * DIM_J * sizeof(elem_t)), 0);
                }

                // printf("[DEBUG][MATMUL] Phase 2.3.3: mmacc_w_b for C00,C01,C10,C11: k = %d\n", k);
                mmacc_w_b(TILE_NUM+0, 2, 0);
                if (has_j1) mmacc_w_b(TILE_NUM+1, 3, 0);
                if (has_i1) mmacc_w_b(TILE_NUM+2, 2, 1);
                if (has_i1 && has_j1) mmacc_w_b(TILE_NUM+3, 3, 1);
                
            }

            msce32_spm(TILE_NUM+0, c00_sp_addr,(int)(J * DIM_J * sizeof(acc_t)), 0);
            if (has_j1) msce32_spm(TILE_NUM+1, c01_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
            if (has_i1) msce32_spm(TILE_NUM+2, c10_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
            if (has_i1 && has_j1) msce32_spm(TILE_NUM+3, c11_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
        }
    }
    msync_spm();

    /* ---------- write back ---------- */
    if (C != NULL) {
      for (size_t i0_in = 0; i0_in < I * DIM_I; i0_in++) {
        for (size_t j = 0; j < J; j += (size_t)C_blocks) {
          // printf("[DEBUG][MATMUL] Phase 3: DMA Store C, i0 = %d, j = %d\n", i0_in, j);
          size_t cols = min_size(J - j, (size_t)C_blocks);
          const elem_t *C_dram = C + i0_in * stride_C + j * DIM_J;  
          uint64_t C_sp = get_C_block_addr(c_current_buffer, i0_in , j, J);
          size_t bytes = cols * DIM_J * sizeof(acc_t);
          uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
          if (cachelines >= 32) cachelines = 0;
          dmastore_spm((uint64_t)C_dram, C_sp, 1);
        }
      }
      msync_spm();
    }

}

/* =============================
 *  Outer tiling
 * ============================= */
static void tiled_matmul_outer(
    size_t dim_I, size_t dim_J, size_t dim_K,
    const elem_t *A, const elem_t *B,
    const acc_t *D, acc_t *C,
    size_t stride_A, size_t stride_B,
    size_t stride_D, size_t stride_C,
    enum tiled_matmul_type_t type,
    size_t tile_I, size_t tile_J, size_t tile_K)
{
    (void)type;
    size_t TI = div_up(dim_I, DIM_I);
    size_t TJ = div_up(dim_J, DIM_J);
    size_t TK = div_up(dim_K, DIM_K);

    size_t I0 = div_up(TI, tile_I);
    size_t J0 = div_up(TJ, tile_J);
    size_t K0 = div_up(TK, tile_K);

    bool no_bias = (D == NULL);
    if (no_bias) D = (acc_t*)1;

    /* 初始化double buffer状态 */
    int current_buffer = 0;
    int next_buffer = 1;
    int c_current_buffer = 0;
    int c_next_buffer = 1;


    int A_blocks = (int)8;
    int B_blocks = (int)(MAX_DMA_BLOCK / (DIM_J * sizeof(elem_t)));
    int D_blocks = (int)(MAX_DMA_BLOCK / (DIM_J * sizeof(acc_t)));
    int C_blocks = (int)(MAX_DMA_BLOCK / (DIM_J * sizeof(acc_t)));
    if (A_blocks < 1) A_blocks = 1;
    if (B_blocks < 1) B_blocks = 1;
    if (D_blocks < 1) D_blocks = 1;
    if (C_blocks < 1) C_blocks = 1;
    if (A_blocks > (int)tile_K) A_blocks = (int)tile_K;
    if (B_blocks > (int)tile_J) B_blocks = (int)tile_J;
    if (D_blocks > (int)tile_J) D_blocks = (int)tile_J;
    if (C_blocks > (int)tile_J) C_blocks = (int)tile_J;
    // printf("[INFO][MATMUL] A_blocks = %d, B_blocks = %d, D_blocks = %d, C_blocks = %d\n", A_blocks, B_blocks, D_blocks, C_blocks);
    /* Move-in D (bias) ：按 tile 行搬入 */
    // printf("[DEBUG][MATMUL] Move-in D (bias)\n");

    /* ---------- bias ---------- */
    if (D != NULL && !no_bias) {
      // printf("[DEBUG][MATMUL] Phase 0: Loading bias to C buffer %d\n", current_buffer);
      for (size_t j = 0; j < tile_J; j += (size_t)D_blocks) {
        for (size_t i0 = 0; i0 < tile_I * DIM_I; i0++) {
          // printf("[DEBUG] Loading initial D for i0 = %d, j = %d\n", i0, j);
          size_t cols = min_size(tile_J - j, (size_t)D_blocks);
          const elem_t *D_dram = D + j * DIM_J;  
          uint64_t D_sp = get_C_block_addr(current_buffer, i0 , j, tile_J);
          size_t bytes = cols * DIM_J * sizeof(elem_t);
          uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
          if (cachelines >= 32) cachelines = 0;
          dmaload_spm((uint64_t)D_dram, D_sp, 1);
        }
      }
    }

    /* 第1阶段：为第一次计算加载初始的A和B数据到buffer0 */
    // printf("[DEBUG][MATMUL] Phase 1: Loading initial A and B data to buffer %d\n", current_buffer);

    /* 加载所有A的初始数据 */
    for (size_t i = 0; i < tile_I; i++) {
      for (size_t k = 0; k < tile_K; k += 1) {
        // printf("[DEBUG] Loading initial A for i = %d, k = %d, K = %d\n", i, k, tile_K);
        const elem_t *A_dram = A + i * DIM_I * stride_A + k * DIM_K * DIM_I;  
        uint64_t A_sp = get_A_dim_block_addr(current_buffer, i, k, tile_K);
        uint8_t cachelines = (uint8_t)div_up(DIM_I * DIM_K * sizeof(elem_t), CACHELINE_SIZE);
        if (cachelines >= 32) cachelines = 0;
        dmaload_spm((uint64_t)A_dram, A_sp, 4); // 搬入一个A的DIM块        
      }
    }
    
    /* 加载所有B的初始数据 */
    for (size_t j = 0; j < tile_J; j += (size_t)B_blocks) {
      for (size_t k0 = 0; k0 < tile_K * DIM_K; k0++) {
        // printf("[DEBUG] Loading initial B for k0 = %d, j = %d\n", k0, j);
        size_t cols = min_size(tile_J - j, (size_t)B_blocks);
        const elem_t *B_dram = B + j * DIM_J + k0 * stride_B;  
        uint64_t B_sp = get_B_block_addr(current_buffer, k0 , j, tile_J);
        size_t bytes = cols * DIM_J * sizeof(elem_t);
        uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
        if (cachelines >= 32) cachelines = 0;
        dmaload_spm((uint64_t)B_dram, B_sp, 1);
      }
    }
    
    msync_spm();

    msettilemi(DIM_I);
    msettileni(DIM_J);
    msettileki(DIM_K);

    for (size_t i0 = 0; i0 < I0; i0++){
      for (size_t j0 = 0; j0 < J0; j0++){
        for (size_t k0 = 0; k0 < K0; k0++) {
          // printf("[DEBUG] OUTSIDE LOOP: Processing i0 = %d, j0 = %d, k0 = %d, D_dramAddr = %d\n", i0, j0, k0, D);
          /* 此外层块内实际 tile 数（尾块可能不足） */
          // const size_t I = (i0 < I0 - 1) ? tile_I : (TI - (I0 - 1) * tile_I);
          // const size_t J = (j0 < J0 - 1) ? tile_J : (TJ - (J0 - 1) * tile_J);
          // const size_t K = (k0 < K0 - 1) ? tile_K : (TK - (K0 - 1) * tile_K);
          size_t I = tile_I;
          size_t J = tile_J;
          size_t K = tile_K;
          const acc_t* pre = NULL, * pre_1 = NULL;
          acc_t* out = NULL;
          if (k0 == 0) {
            pre = (acc_t*)D + (i0 * tile_I * DIM_I) * stride_D + j0 * tile_J * DIM_J;
          }
          if (k0 == K0 - 1) {
            out = (acc_t*)C + (i0 * tile_I * DIM_I) * stride_C + j0 * tile_J * DIM_J;
          }
          const elem_t *a = A + (i0 * tile_I * DIM_I) * stride_A + (k0 * tile_K * DIM_K * DIM_I);
          const elem_t *b = B + (k0 * tile_K * DIM_K) * stride_B + (j0 * tile_J * DIM_J);
          // printf("[INFO][MATMUL] Outer tile block (%zu, %zu, %zu): I=%zu, J=%zu, K=%zu, pre=%p, out=%p\n",
          //     i0, j0, k0, I, J, K, pre, out);
          /* 传入 OS 内核：row_stride 以“元素”为单位 */

          const elem_t *a1, *b1;


          if (k0 == K0 - 1){
            if(j0 == J0 -1){
              if (i0 == I0 - 1) {
                a1 = A;
              } else {
                a1 = a + tile_I * DIM_I * stride_A - (K0 - 1) * tile_K * DIM_K * DIM_I;
              }
            } else {
              a1 = a - (K0 -1) * tile_K * DIM_K * DIM_I; // 回退到这行的开头
            }
          } else {
              a1 = a + (tile_K * DIM_K * DIM_I);
          }
        
          if (k0 == K0 -1){
            if (j0 == J0 - 1) {
              if (i0 == I0 - 1) {
                b1 = B;
              } else {
                b1 = B;
              }
            } else {
              b1 = b + tile_J * DIM_J - (K0 - 1) * tile_K * DIM_K * stride_B;
            }
          } else {
            b1 = b + (stride_B * tile_K * DIM_K);
          }
          
          if (k0 == K0 - 1) {
            if(j0 == J0 - 1){
              if (i0 == I0 - 1){
                pre_1 = NULL;
              } else {
                pre_1 = (acc_t*)D + (i0 * tile_I * DIM_I) * stride_D + j0 * tile_J * DIM_J + tile_J * DIM_J;
              }
            } else{
              pre_1 = (acc_t*)D + (i0 * tile_I * DIM_I) * stride_D + (j0 + 1) * tile_J * DIM_J;
            }
          }

          // printf("[DEBUG] OUTSIDE LOOP: a1 = %d, b1 = %d, pre_1 = %d, out = %d\n", a1, b1, pre_1, out);

          if (k0 == K0 - 1) {
            pre_1 = (acc_t*)D + (i0 * tile_I * DIM_I) * stride_D + j0 * tile_J * DIM_J + tile_J * DIM_J;
          }
          sp_tiled_matmul_os(a, b, D, out,
              I, J, K,
              stride_A, stride_B,
              stride_D, stride_C,
              pre_1 == (acc_t*)1,
              current_buffer, next_buffer,
              c_current_buffer, c_next_buffer,
              i0, j0, k0,
              I0, J0, K0,
              a1, b1);

          int t = current_buffer; current_buffer = next_buffer; next_buffer = t;
        }
        int t_c = c_current_buffer; c_current_buffer = c_next_buffer; c_next_buffer = t_c;
      }
    }
}

/* =============================
 *  tiled_matmul
 * ============================= */
static void tiled_matmul(
    size_t dim_I, size_t dim_J, size_t dim_K,
    const elem_t *A, const elem_t *B,
    const acc_t *D, acc_t *C,
    size_t stride_A, size_t stride_B,
    size_t stride_D, size_t stride_C,
    enum tiled_matmul_type_t type,
    size_t tile_I, size_t tile_J, size_t tile_K)
{
#if ASSERTIONS
    if (!is_spad_rows_satisfied_for_one_buffer(tile_I, tile_J, tile_K)) {
        // printf("SPM overflow in double buffer mode\n");
        exit(1);
    }
#endif
    tiled_matmul_outer(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B,
        stride_D, stride_C,
        type, tile_I, tile_J, tile_K);
}

// dim_xxx 以"元素"为单位，代表完整矩阵的实际维度
// C = A * B + D，其中D与C元素尺寸相同
// stride_xxx 以"元素"为单位
static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
    const elem_t* A, const elem_t* B, const acc_t* D, acc_t* C,
    size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C) {

  asm volatile ("" ::: "memory");
  asm volatile ("fence rw, rw" ::: "memory");

  /* 使用double buffer的单buffer容量计算最大tile_ij */
  const size_t max_tile_ij_in_bank_c = ((size_t)sqrt(BANK_C_BUF_ROWS * BANK_WIDTH / (DIM_I * DIM_J) / sizeof(acc_t)));
  const size_t dim_I_padded = (dim_I / DIM_I + (dim_I % DIM_I != 0)) * DIM_I;
  const size_t dim_J_padded = (dim_J / DIM_J + (dim_J % DIM_J != 0)) * DIM_J;
  const size_t dim_K_padded = (dim_K / DIM_K + (dim_K % DIM_K != 0)) * DIM_K;
  size_t tile_I = dim_I_padded / DIM_I;
  size_t tile_J = dim_J_padded / DIM_J;
  size_t tile_K = dim_K_padded / DIM_K;
  // printf("[INFO][MATMUL] Double Buffer mode\n");
  // printf("[INFO][MATMUL] max_tile_ij_in_bank_c = %d, tile_I = %d, tile_J = %d, tile_K = %d\n", 
    // (int)max_tile_ij_in_bank_c, (int)tile_I, (int)tile_J, (int)tile_K);
  enum tiled_matmul_type_t tiled_matmul_type;
  
  // tiled_matmul_type = OS;
  //   tile_I = 2;
  //   tile_J = 8;
  //   tile_K = 2;

  // /* 贪心扩张 */
  // while (true) {
  //   bool increased = false;
  //   if (is_spad_rows_satisfied_for_one_buffer(tile_I, tile_J, tile_K + 2) &&
  //       (tile_K + 2) * DIM_K <= dim_K_padded) {
  //     tile_K+=2;
  //     increased = true;
  //     // printf("[DEBUG][MATMUL] Increase K to %d\n", (int)tile_K);
  //     continue;
  //   }
  //   if (is_spad_rows_satisfied_for_one_buffer(tile_I, tile_J + 8, tile_K) &&
  //       (tile_J + 8) * DIM_J <= dim_J_padded) {
  //     tile_J+=8;
  //     increased = true;
  //     // printf("[DEBUG][MATMUL] Increase J to %d\n", (int)tile_J);
  //     continue;
  //   }
  //   if (is_spad_rows_satisfied_for_one_buffer(tile_I + 2, tile_J, tile_K) &&
  //       (tile_I + 2) * DIM_I <= dim_I_padded) {
  //     tile_I+=2;
  //     increased = true;
  //     // printf("[DEBUG][MATMUL] Increase I to %d\n", (int)tile_I);
  //     continue;
  //   }
  //   if (!increased) break;
  // }

    tiled_matmul_type = OS;
    tile_I = 8;
    tile_J = 8;
    tile_K = 8;

  // printf("[INFO][MATMUL] Auto tiling selected for double buffer: I=%d, J=%d, K=%d, type=%s\n",
  //     (int)tile_I, (int)tile_J, (int)tile_K,
  //     tiled_matmul_type == OS ? "OS" : (tiled_matmul_type == WS ? "WS" : "CPU"));
#if PRINT_TILE
  {
    const int bank_a_rows = (tile_I * DIM_I * tile_K * DIM_K) * sizeof(elem_t) / BANK_WIDTH;
    const int bank_b_rows = (tile_J * DIM_J * tile_K * DIM_K) * sizeof(elem_t) / BANK_WIDTH;
    const int bank_c_rows = (tile_I * DIM_I * tile_J * DIM_J) * sizeof(acc_t) / BANK_WIDTH;
    // printf("[INFO][MATMUL] tile_I: %d\n", (int)tile_I);
    // printf("[INFO][MATMUL] tile_J: %d\n", (int)tile_J);
    // printf("[INFO][MATMUL] tile_K: %d\n\n", (int)tile_K);
    // printf("[INFO][MATMUL] bankA_used_rows per buffer: %d (max %d)\n", bank_a_rows, BANK_A_BUF_ROWS);
    // printf("[INFO][MATMUL] bankB_used_rows per buffer: %d (max %d)\n", bank_b_rows, BANK_B_BUF_ROWS);
    // printf("[INFO][MATMUL] bankC_used_rows per buffer: %d (max %d)\n", bank_c_rows, BANK_C_BUF_ROWS);
    // printf("[INFO][MATMUL] Bank A utilization per buffer: %f%%\n", (float)(bank_a_rows * 100) / BANK_A_BUF_ROWS);
    // printf("[INFO][MATMUL] Bank B utilization per buffer: %f%%\n", (float)(bank_b_rows * 100) / BANK_B_BUF_ROWS);
    // printf("[INFO][MATMUL] Bank C utilization per buffer: %f%%\n\n", (float)(bank_c_rows * 100) / BANK_C_BUF_ROWS);
  }
  // exit(1);
#endif
  /* 直接走 OS 实现（当前文件只实现了 OS 内核） */
  (void)tiled_matmul_type; /* 防止未使用告警 */
  tiled_matmul(dim_I, dim_J, dim_K,
      A, B, D, C, stride_A, stride_B, stride_D, stride_C,
      OS, tile_I, tile_J, tile_K);

  asm volatile ("" ::: "memory");
  asm volatile ("fence rw, rw" ::: "memory");
}

#endif
