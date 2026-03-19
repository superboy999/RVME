#ifndef __MATMUL_H__
#define __MATMUL_H__

#include "inst.h"
#include "config_2.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

// long long matmul_cnt = 0;

// void add_matmul_cnt(long long x) {
//   matmul_cnt = matmul_cnt + x;
// }

static void sp_tiled_matmul_os(const elem_t * A, const elem_t * B, const acc_t * D, acc_t * C,
        size_t I, size_t J, size_t K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool no_bias, elem_t dim_K_lt_256) {
  /* SPM 统一使用“字节地址” */
  const uint64_t A_sp_addr_start = 0;
  const uint64_t B_sp_addr_start = BANK_A_ROWS * BANK_WIDTH;
  const uint64_t D_sp_addr_start = BANK_A_ROWS * BANK_WIDTH + BANK_B_ROWS * BANK_WIDTH;
  const uint64_t C_sp_addr_start = D_sp_addr_start; /* 共享同一块 */
  /* 每次 DMA 最大传输按字节限制换算成 tile 列块数 */
  // int A_blocks = (int)(MAX_DMA_BLOCK / (DIM_K * sizeof(elem_t)));
  // int B_blocks = (int)(MAX_DMA_BLOCK / (DIM_J * sizeof(elem_t)));
  // int D_blocks = (int)(MAX_DMA_BLOCK / (DIM_J * sizeof(acc_t)));
  // int C_blocks = (int)(MAX_DMA_BLOCK / (DIM_J * sizeof(acc_t)));
  int A_blocks = (int)(MAX_DMA_BLOCK / (DIM_K * sizeof(elem_t)));
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
  // printf("[INFO][MATMUL] Start sp_tiled_matmul_os(): I=%zu, J=%zu, K=%zu\n", I, J, K);
  // printf("[INFO][MATMUL] A_blocks = %d, B_blocks = %d, D_blocks = %d, C_blocks = %d\n", A_blocks, B_blocks, D_blocks, C_blocks);
  /* Move-in D (bias) ：按 tile 行搬入 */
  // printf("[DEBUG][MATMUL] Move-in D (bias)\n");
  if (D != NULL && !no_bias) {
    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j += (size_t)D_blocks) {
        size_t cols = min_size(J - j, (size_t)D_blocks);
        for (size_t i0 = 0; i0 < DIM_I; i0++) {
          size_t bias_row = i * DIM_I + i0;
          const acc_t *D_dram_addr = D + bias_row * D_row_stride + j * DIM_J;
          uint64_t D_sp_addr = D_sp_addr_start
              + (uint64_t)((bias_row * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
          // printf("D = %zu, bias_row = %zu, D_row_stride = %zu, j = %zu, DIM_J = %zu \n ", D, bias_row, D_row_stride, j, DIM_J);
          size_t bytes = cols * DIM_J * sizeof(acc_t);
          // printf("[DEBUG][MATMUL] Move-in bias: D[%zu, %zu]-[%zu, %zu] (%zu bytes) from Mem addr 0x%lx-0x%lx to SPM addr 0x%lx-0x%lx\n",
          //     bias_row, j * DIM_J, bias_row, (j + cols) * DIM_J - 1, bytes, D_dram_addr, D_dram_addr + bytes - 1, D_sp_addr, D_sp_addr + bytes - 1);
          uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
          if (cachelines>=32) cachelines = 0; /* uimm5 */
          dmaload_spm((uint64_t)D_dram_addr, D_sp_addr, 1);
        }
      }
    }
  }
  /* Move-in B ：按行搬入，优先搬完完整的一行，再按k方向搬 */
  // printf("[DEBUG][MATMUL] Move-in B\n");
  for (size_t k0 = 0; k0 < K * DIM_K; k0++) {
    for (size_t j = 0; j < J; j += (size_t)B_blocks) {
      // printf("[DEBUG] load B: k0 = %d, j = %d\n", k0, j);
      size_t cols = min_size(J - j, (size_t)B_blocks);
      const elem_t *B_dram_addr = B + k0 * B_row_stride + j * DIM_J;
      uint64_t B_sp_addr = B_sp_addr_start + (uint64_t)((k0 * (J * DIM_J) + j * DIM_J) * sizeof(elem_t));
      size_t bytes = cols * DIM_J * sizeof(elem_t);
      // printf("[DEBUG][MATMUL] Move-in B: B[%zu, %zu]-[%zu, %zu] (%zu bytes) from Mem addr 0x%lx-0x%lx to SPM addr 0x%lx-0x%lx\n",
      //       k0, j * DIM_J, k0, (j + cols) * DIM_J - 1, bytes, B_dram_addr, B_dram_addr + bytes - 1, B_sp_addr, B_sp_addr + bytes - 1);
      uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
      if (cachelines>=32) cachelines = 0; /* uimm5 */
      dmaload_spm((uint64_t)B_dram_addr, B_sp_addr, 1);
    }
  }
  /* Move-in A ：按register块搬入，一次搬入完整的一个tile寄存器块，再按k方向搬，最后按i方向搬 */
  // printf("[DEBUG][MATMUL] Move-in A\n");
  if (! dim_K_lt_256){
    for (size_t i = 0; i < I; i++) {
      for (size_t k = 0; k < K; k++) {
        // printf("[DEBUG] load A: i = %d, k = %d\n", i, k);
        const elem_t *A_dram_addr = A + i * DIM_I * A_row_stride + k * DIM_K * DIM_I;
        uint64_t A_sp_addr = A_sp_addr_start + (uint64_t)((i * (K * DIM_K * DIM_I) + k * DIM_K * DIM_I) * sizeof(elem_t));
        size_t bytes = DIM_I * DIM_K * sizeof(elem_t);
        // printf("[DEBUG][MATMUL] Move-in A: A[%zu, %zu]-[%zu, %zu] (%zu bytes) from Mem addr 0x%lx-0x%lx to SPM addr 0x%lx-0x%lx\n",
        //     i*DIM_I, k * DIM_K, (i+1)*DIM_I-1, (k+1)*DIM_K - 1, bytes, A_dram_addr, A_dram_addr + bytes - 1, A_sp_addr, A_sp_addr + bytes - 1);
        uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
        if (cachelines>=32) cachelines = 0; /* uimm5 */
        dmaload_spm((uint64_t)A_dram_addr, A_sp_addr, 4);        
      }
    }
  } else {
    for (size_t i = 0; i < I; i++) {
      for (size_t k = 0; k < K; k++) {
        // printf("[DEBUG] load A: i = %d, k = %d\n", i, k);
        const elem_t *A_dram_addr = A + i * 2 * A_row_stride + k * A_row_stride;
        uint64_t A_sp_addr = A_sp_addr_start + (uint64_t)((i * (K * DIM_K * DIM_I) + k * DIM_K * DIM_I) * sizeof(elem_t));
        size_t bytes = DIM_I * DIM_K * sizeof(elem_t);
        // printf("[DEBUG][MATMUL] Move-in A: A[%zu, %zu]-[%zu, %zu] (%zu bytes) from Mem addr 0x%lx-0x%lx to SPM addr 0x%lx-0x%lx\n",
        //     i*DIM_I, k * DIM_K, (i+1)*DIM_I-1, (k+1)*DIM_K - 1, bytes, A_dram_addr, A_dram_addr + bytes - 1, A_sp_addr, A_sp_addr + bytes - 1);
        uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
        if (cachelines>=32) cachelines = 0; /* uimm5 */
        dmaload_spm((uint64_t)A_dram_addr, A_sp_addr, 4);     
        // dmaload_spm((uint64_t)A_dram_addr + A_row_stride, A_sp_addr + 64, 4); 
        // dmaload_spm((uint64_t)A_dram_addr + 2 * A_row_stride, A_sp_addr + 2 * 64, 4); 
        // dmaload_spm((uint64_t)A_dram_addr + 3 * A_row_stride, A_sp_addr + 3 * 64, 4); 
      }
    }
  }
  msync_spm();
  /* Compute */
  // printf("[DEBUG][MATMUL] Compute\n");
  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j++) {
      uint64_t C_sp_addr = C_sp_addr_start
          + (uint64_t)(((i * DIM_I) * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
      msettilemi(DIM_I);
      msettileni(DIM_J);
      // printf("[DEBUG] compute: i = %d, j = %d\n", i, j);
      /* acc0 初值：有 bias -> 载入 bias；无 bias -> 保证为零 */
      if (!no_bias) {
        /* stride 以“Byte”为单位，传整块行跨度 */
        // printf("[DEBUG] I = %d, J =%d, stride in bytes = %zu\n", (int)I, (int)J, (size_t)(J * DIM_J * sizeof(acc_t)));
        // printf("[DEBUG] Load-in bias to acc0[%zu, %zu]: from SPM addr 0x%lx\n",
        //   i * DIM_I, j * DIM_J, C_sp_addr);
        mlce32_spm(TILE_NUM+0, C_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
      } else {
        // /* 若无专用清零指令，则将零 tile 从 DRAM 搬到 SPM 再载入（保守实现） */
        // size_t bytes = DIM_I * DIM_J * sizeof(acc_t);
        // // printf("[DEBUG] Clear acc0[%zu, %zu]: zero tile from DRAM to SPM addr 0x%lx\n",
        // //   i * DIM_I, j * DIM_J, C_sp_addr);
        // uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
        // if (cachelines>=32) cachelines = 0; /* uimm5 */
        // dmaload_spm((uint64_t)g_zero_tile, C_sp_addr, cachelines);
        // mlce32_spm(TILE_NUM+0, C_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);

        /* 暂时用msub指令代替mzero指令清零 */
        // mlce32_spm(TILE_NUM+0, C_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
        mzero(TILE_NUM+0);
        // msub_w_mm(TILE_NUM+0, TILE_NUM+0, TILE_NUM+0);
      }
      msettileki(DIM_K);
      for (size_t k = 0; k < K; k++) {
        uint64_t A_sp_addr;
        if (! dim_K_lt_256) {
          A_sp_addr = A_sp_addr_start
              + (uint64_t)(((i * DIM_I) * (K * DIM_K) + k * DIM_K * DIM_I) * sizeof(elem_t));          
        } else {
          A_sp_addr = A_sp_addr_start
              + (uint64_t)(((i * DIM_I) * (K * DIM_K) + k * DIM_K * DIM_I) * sizeof(elem_t));       
        }

        uint64_t B_sp_addr = B_sp_addr_start
            + (uint64_t)(((k * DIM_K) * (J * DIM_J) + j * DIM_J) * sizeof(elem_t));
        /* 这里第三参为“行跨度（Byte数）”，而非 tile 宽度 */
        mlae8_spm(0, A_sp_addr, (int)(DIM_K * sizeof(elem_t)), 0); /* tile0 <- A(i,k) */
        mlbe8_spm(2, B_sp_addr, (int)(J * DIM_J * sizeof(elem_t)), 0); /* tile2 <- B(k,j) */
        mmacc_w_b(TILE_NUM+0, 2, 0); /* acc0 += tile0 * tile2 */
        // add_matmul_cnt(4160); // 8 * 32 * 8 * 2 + 8 * 8
        
      }
      msce32_spm(TILE_NUM+0, C_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0); /* acc0 -> C tile (写回 SPM) */
    }
  }
  msync_spm();
  /* Move-out C ：按行搬回 DRAM */
  // printf("[DEBUG][MATMUL] Move-out C\n");
  if (C != NULL) {
    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j += (size_t)C_blocks) {
        // printf("[DEBUG] dma store C\n");
        size_t cols = min_size(J - j, (size_t)C_blocks);
        // printf("[DEBUG][MATMUL] cols = min(%zu, %zu) = %zu\n", J - j, (size_t)C_blocks, cols);
        for (size_t i0 = 0; i0 < DIM_I; i0++) {
          size_t c_row = i * DIM_I + i0;
          acc_t *C_dram_addr = C + c_row * C_row_stride + j * DIM_J;
          uint64_t C_sp_addr = C_sp_addr_start
              + (uint64_t)((c_row * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
          size_t bytes = cols * DIM_J * sizeof(acc_t);
          // printf("[DEBUG][MATMUL] Move-out C[%zu, %zu]-[%zu, %zu] (%zu bytes) from SPM addr 0x%lx-0x%lx to Mem addr 0x%lx-0x%lx\n",
          //     c_row, j * DIM_J, c_row, (j + cols) * DIM_J - 1, bytes, C_sp_addr, C_sp_addr + bytes - 1, C_dram_addr, C_dram_addr + bytes - 1);
          uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
          if (cachelines>=32) cachelines = 0; /* uimm5 */
          dmastore_spm((uint64_t)C_dram_addr, C_sp_addr, 1);
        }
      }
    }
  }
}


static void tiled_matmul_outer(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B, const acc_t * D, acc_t * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        enum tiled_matmul_type_t tiled_matmul_type,
        size_t tile_I, size_t tile_J, size_t tile_K) {
  (void)tiled_matmul_type; /* 仅实现 OS */
  /* 元素维度转 tile 维度（向上取整） */
  const size_t TI = div_up(dim_I, (size_t)DIM_I);
  const size_t TJ = div_up(dim_J, (size_t)DIM_J);
  const size_t TK = div_up(dim_K, (size_t)DIM_K);
  /* 外层块数量（每块包含 tile_I/J/K 个 tile） */
  const size_t I0 = div_up(TI, tile_I);
  const size_t J0 = div_up(TJ, tile_J);
  const size_t K0 = div_up(TK, tile_K);
  const bool no_bias = (D == NULL);
  if (no_bias) D = (acc_t*)1; /* 占位非空 */
  for (size_t i0 = 0; i0 < I0; i0++) {
    for (size_t j0 = 0; j0 < J0; j0++) {
      for (size_t k0 = 0; k0 < K0; k0++) {
        /* 此外层块内实际 tile 数（尾块可能不足） */
        const size_t I = (i0 < I0 - 1) ? tile_I : (TI - (I0 - 1) * tile_I);
        const size_t J = (j0 < J0 - 1) ? tile_J : (TJ - (J0 - 1) * tile_J);
        const size_t K = (k0 < K0 - 1) ? tile_K : (TK - (K0 - 1) * tile_K);
        const acc_t* pre = NULL;
        acc_t* out = NULL;
        if (k0 == 0 && !no_bias) {
          pre = (acc_t*)D + (i0 * tile_I * DIM_I) * stride_D + j0 * tile_J * DIM_J;
        }
        if (k0 == K0 - 1) {
          out = (acc_t*)C + (i0 * tile_I * DIM_I) * stride_C + j0 * tile_J * DIM_J;
        }
        const elem_t *a = A + (i0 * tile_I * DIM_I) * stride_A + (k0 * tile_K * DIM_K * DIM_I);
        const elem_t *b = B + (k0 * tile_K * DIM_K) * stride_B + (j0 * tile_J * DIM_J);
        // printf("[INFO][MATMUL] Outer tile block (%zu, %zu, %zu): I=%zu, J=%zu, K=%zu, pre=%p, out=%p\n",
        //     i0, j0, k0, I, J, K, pre, out);
        elem_t dim_K_lt_256 = (elem_t)(dim_K < 256);
        /* 传入 OS 内核：row_stride 以“元素”为单位 */
        sp_tiled_matmul_os(a, b, pre, out,
            I, J, K, stride_A, stride_B, stride_D, stride_C, (/*pre == (acc_t*)1*/no_bias), dim_K_lt_256);
      }
    }
  }
}


static void tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B, const acc_t* D, acc_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        enum tiled_matmul_type_t tiled_matmul_type,
        size_t tile_I, size_t tile_J, size_t tile_K) {
#if ASSERTIONS
  if (tile_I == 0 || tile_J == 0 || tile_K == 0) {
    // printf("tile factors must be positive\n");
    exit(1);
  }
  const size_t dim_I_padded = (dim_I / DIM_I + (dim_I % DIM_I != 0)) * DIM_I;
  const size_t dim_J_padded = (dim_J / DIM_J + (dim_J % DIM_J != 0)) * DIM_J;
  const size_t dim_K_padded = (dim_K / DIM_K + (dim_K % DIM_K != 0)) * DIM_K;
  if (tile_I * DIM_I > dim_I_padded) {
    // printf("tile_I too large\n"); exit(1);
  }
  if (tile_J * DIM_J > dim_J_padded) {
    // printf("tile_J too large\n"); exit(1);
  }
  if (tile_K * DIM_K > dim_K_padded) {
    // printf("tile_K too large\n"); exit(1);
  }
  const int bank_a_rows = (tile_I * DIM_I * tile_K * DIM_K) / BANK_WIDTH;
  const int bank_b_rows = (tile_J * DIM_J * tile_K * DIM_K) / BANK_WIDTH;
  const int bank_c_rows = (tile_I * DIM_I * tile_J * DIM_J) / BANK_WIDTH;
  if (bank_a_rows > BANK_A_ROWS) { printf("Bank A over-utilized\n"); exit(1); }
  if (bank_b_rows > BANK_B_ROWS) { printf("Bank B over-utilized\n"); exit(1); }
  if (bank_c_rows > BANK_C_ROWS) { printf("Bank C over-utilized\n"); exit(1); }
  if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535) {
    // printf("I/J/K tiling factors must be < 65535\n"); exit(1);
  }
#endif
  tiled_matmul_outer(dim_I, dim_J, dim_K,
      A, B, D, C, stride_A, stride_B, stride_D, stride_C,
      tiled_matmul_type, tile_I, tile_J, tile_K);
}


// dim_xxx 以“元素”为单位，代表完整矩阵的实际维度
// C = A * B + D，其中D与C元素尺寸相同
// stride_xxx 以“元素”为单位
static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
    const elem_t* A, const elem_t* B, const acc_t* D, acc_t* C,
    size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C) {

  asm volatile ("" ::: "memory");
  asm volatile ("fence rw, rw" ::: "memory");

  const size_t max_tile_ij_in_bank_c = ((size_t)sqrt(BANK_C_ROWS * BANK_WIDTH / (DIM_I * DIM_J) / sizeof(acc_t)));
  const size_t dim_I_padded = (dim_I / DIM_I + (dim_I % DIM_I != 0)) * DIM_I;
  const size_t dim_J_padded = (dim_J / DIM_J + (dim_J % DIM_J != 0)) * DIM_J;
  const size_t dim_K_padded = (dim_K / DIM_K + (dim_K % DIM_K != 0)) * DIM_K;
  size_t tile_I = dim_I_padded / DIM_I;
  size_t tile_J = dim_J_padded / DIM_J;
  size_t tile_K = dim_K_padded / DIM_K;
  // printf("[INFO][MATMUL] max_tile_ij_in_bank_c = %d, tile_I = %d, tile_J = %d, tile_K = %d\n", 
    // (int)max_tile_ij_in_bank_c, (int)tile_I, (int)tile_J, (int)tile_K);
  enum tiled_matmul_type_t tiled_matmul_type;
  /* 策略：若 B bank 能容纳整块B矩阵（I=1），则 WS，否则 OS（保持与原有策略一致） */
//   if (is_spad_rows_satisfied(1, tile_J, tile_K)) {
//     tiled_matmul_type = WS;
//     tile_I = 1;
//   } else {
    tiled_matmul_type = OS;
    // tile_I = tile_I < max_tile_ij_in_bank_c ? tile_I : max_tile_ij_in_bank_c;
    // tile_J = tile_J < max_tile_ij_in_bank_c ? tile_J : max_tile_ij_in_bank_c;
    if (dim_K < 256){
      tile_I = 2;
      tile_J = 8;
      tile_K = 2;
    } else {
      tile_I = 16;
      tile_J = 16;
      tile_K = 8;
    }

//   }
  
  /* 贪心扩张 */
  while (true) {
    bool increased = false;
    if (is_spad_rows_satisfied(tile_I, tile_J, tile_K + 2) &&
        (tile_K + 2) * DIM_K <= dim_K_padded) {
      tile_K+=2;
      increased = true;
      // printf("[DEBUG][MATMUL] Increase K to %d\n", (int)tile_K);
      continue;
    }
    if (is_spad_rows_satisfied(tile_I, tile_J + 8, tile_K) &&
        (tile_J + 8) * DIM_J <= dim_J_padded) {
      tile_J+=8;
      increased = true;
      // printf("[DEBUG][MATMUL] Increase J to %d\n", (int)tile_J);
      continue;
    }
    if (is_spad_rows_satisfied(tile_I + 2, tile_J, tile_K) &&
        (tile_I + 2) * DIM_I <= dim_I_padded) {
      tile_I+=2;
      increased = true;
      // printf("[DEBUG][MATMUL] Increase I to %d\n", (int)tile_I);
      continue;
    }
    if (!increased) break;
  }
  // printf("[INFO][MATMUL] Auto tiling selected: I=%d, J=%d, K=%d, type=%s\n",
      // (int)tile_I, (int)tile_J, (int)tile_K,
      // tiled_matmul_type == OS ? "OS" : (tiled_matmul_type == WS ? "WS" : "CPU"));
#if PRINT_TILE
  {
    const int bank_a_rows = (tile_I * DIM_I * tile_K * DIM_K) * sizeof(elem_t) / BANK_WIDTH;
    const int bank_b_rows = (tile_J * DIM_J * tile_K * DIM_K) * sizeof(elem_t) / BANK_WIDTH;
    const int bank_c_rows = (tile_I * DIM_I * tile_J * DIM_J) * sizeof(acc_t) / BANK_WIDTH;
    printf("[INFO][MATMUL] tile_I: %d\n", (int)tile_I);
    printf("[INFO][MATMUL] tile_J: %d\n", (int)tile_J);
    printf("[INFO][MATMUL] tile_K: %d\n\n", (int)tile_K);
    printf("[INFO][MATMUL] bankA_used_rows: %d\n", bank_a_rows);
    printf("[INFO][MATMUL] bankB_used_rows: %d\n", bank_b_rows);
    printf("[INFO][MATMUL] bankC_used_rows: %d\n", bank_c_rows);
    printf("[INFO][MATMUL] Bank A utilization: %f%%\n", (float)(bank_a_rows * 100) / BANK_A_ROWS);
    printf("[INFO][MATMUL] Bank B utilization: %f%%\n", (float)(bank_b_rows * 100) / BANK_B_ROWS);
    printf("[INFO][MATMUL] Bank C utilization: %f%%\n\n", (float)(bank_c_rows * 100) / BANK_C_ROWS);
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