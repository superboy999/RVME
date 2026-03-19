#ifndef __MATADD_H__
#define __MATADD_H__

#include "inst.h"
#include "config_v2.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdalign.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

long long matadd_cnt = 0;

void add_matadd_cnt(long long x) {
  matadd_cnt = matadd_cnt + x;
}

static void sp_tiled_add(const acc_t* in1, const acc_t* in2, acc_t* out,
        size_t I, size_t J, size_t in1_row_stride, size_t in2_row_stride, size_t out_row_stride) {
  const uint64_t C1_sp_addr_start = BANK_A_ROWS * BANK_WIDTH + BANK_B_ROWS * BANK_WIDTH;
  const uint64_t C2_sp_addr_start = BANK_A_ROWS * BANK_WIDTH + BANK_B_ROWS * BANK_WIDTH + BANK_C_ROWS * BANK_WIDTH / 2;
  /* 每次 DMA 最大传输按字节限制换算成 tile 列块数 */
  int C_blocks = (int)(MAX_DMA_BLOCK / (DIM_J * sizeof(acc_t)));
  if (C_blocks < 1) C_blocks = 1;
  if (C_blocks > (int)J) C_blocks = (int)J;
  // printf("[INFO][ADD] Start sp_tiled_add(): I=%zu, J=%zu, C_blocks=%d\n", I, J, C_blocks);
  // printf("[DEBUG][ADD] Move-in first input matrix in1\n");
  for (size_t i0 = 0; i0 < I * DIM_I; i0++) {
    for (size_t j = 0; j < J; j += (size_t)C_blocks) {
      size_t cols = min_size(J - j, (size_t)C_blocks);
      const acc_t *input_dram_addr = in1 + i0 * in1_row_stride + j * DIM_J;
      uint64_t input_sp_addr = C1_sp_addr_start
          + (uint64_t)((i0 * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
      // size_t bytes = cols * DIM_J * sizeof(acc_t);
      // printf("[DEBUG][ADD] Move-in input: Input[%zu, %zu]-[%zu, %zu] (%zu bytes) from Mem addr 0x%lx-0x%lx to SPM addr 0x%lx-0x%lx\n",
      //     i0, j * DIM_J, i0, (j + cols) * DIM_J - 1, bytes, input_dram_addr, input_dram_addr + bytes - 1, input_sp_addr, input_sp_addr + bytes - 1);
      // uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
      // if (cachelines>=32) cachelines = 0; /* uimm5 */
      dmaload_spm((uint64_t)input_dram_addr, input_sp_addr, 1);
    }
  }
  // printf("[DEBUG][ADD] Move-in second input matrix in2\n");
  for (size_t i0 = 0; i0 < I * DIM_I; i0++) {
    for (size_t j = 0; j < J; j += (size_t)C_blocks) {
      size_t cols = min_size(J - j, (size_t)C_blocks);
      const acc_t *input_dram_addr = in2 + i0 * in2_row_stride + j * DIM_J;
      uint64_t input_sp_addr = C2_sp_addr_start
          + (uint64_t)((i0 * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
      // size_t bytes = cols * DIM_J * sizeof(acc_t);
      // printf("[DEBUG][ADD] Move-in input: Input[%zu, %zu]-[%zu, %zu] (%zu bytes) from Mem addr 0x%lx-0x%lx to SPM addr 0x%lx-0x%lx\n",
      //     i0, j * DIM_J, i0, (j + cols) * DIM_J - 1, bytes, input_dram_addr, input_dram_addr + bytes - 1, input_sp_addr, input_sp_addr + bytes - 1);
      // uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
      // if (cachelines>=32) cachelines = 0; /* uimm5 */
      dmaload_spm((uint64_t)input_dram_addr, input_sp_addr, 1);
    }
  }
  msync_spm();
  // printf("[DEBUG][ADD] Start computing C = in1 + in2\n");
  for (size_t i = 0; i < I; i++) {
    msettilemi(DIM_I);
    for (size_t j = 0; j < J; j++) {
        msettileni(DIM_J);
        uint64_t input1_sp_addr = C1_sp_addr_start
            + (uint64_t)(((i * DIM_I) * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        mlce32_spm(TILE_NUM+0, input1_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
        uint64_t input2_sp_addr = C2_sp_addr_start
            + (uint64_t)(((i * DIM_I) * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        mlce32_spm(TILE_NUM+1, input2_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
        madd_w_mm(TILE_NUM+2, TILE_NUM+0, TILE_NUM+1); /* acc2 = acc0 + acc1 */
        add_matadd_cnt(8 * 8);
        msce32_spm(TILE_NUM+2, input1_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
    }
  }
  msync_spm();
  // printf("[DEBUG][ADD] Start moving out result matrix C from SPM to Mem\n");
  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j += (size_t)C_blocks) {
      size_t cols = min_size(J - j, (size_t)C_blocks);
      for (size_t i0 = 0; i0 < DIM_I; i0++) {
        size_t row = i * DIM_I + i0;
        acc_t *out_dram_addr = out + row * out_row_stride + j * DIM_J;
        uint64_t C_sp_addr = C1_sp_addr_start
            + (uint64_t)((row * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        // size_t bytes = cols * DIM_J * sizeof(acc_t);
        // printf("[DEBUG][ADD] Move-out C[%zu, %zu]-[%zu, %zu] (%zu bytes) from SPM addr 0x%lx-0x%lx to Mem addr 0x%lx-0x%lx\n",
        //     row, j * DIM_J, row, (j + cols) * DIM_J - 1, bytes, C_sp_addr, C_sp_addr + bytes - 1, out_dram_addr, out_dram_addr + bytes - 1);
        // uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
        // if (cachelines>=32) cachelines = 0; /* uimm5 */
        dmastore_spm((uint64_t)out_dram_addr, C_sp_addr, 1);
      }
    }
  }
}


static void tiled_add(size_t dim_I, size_t dim_J, const acc_t* in1, const acc_t* in2, acc_t* out,
    size_t stride_in1, size_t stride_in2, size_t stride_out, size_t tile_I, size_t tile_J) {
#if ASSERTIONS
  if (tile_I == 0 || tile_J == 0) {
    // printf("tile factors must be positive\n");
    exit(1);
  }
  const size_t dim_I_padded = (dim_I / DIM_I + (dim_I % DIM_I != 0)) * DIM_I;
  const size_t dim_J_padded = (dim_J / DIM_J + (dim_J % DIM_J != 0)) * DIM_J;
  if (tile_I * DIM_I > dim_I_padded) {
    // printf("tile_I too large\n"); exit(1);
  }
  if (tile_J * DIM_J > dim_J_padded) {
    // printf("tile_J too large\n"); exit(1);
  }
  const int bank_c_rows = (tile_I * DIM_I * tile_J * DIM_J) / BANK_WIDTH;
  if (bank_c_rows > BANK_C_ROWS) { printf("Bank C over-utilized\n"); exit(1); }
  if (tile_I > 65535 || tile_J > 65535) {
    // printf("I/J tiling factors must be < 65535\n"); exit(1);
  }
#endif
  asm volatile ("" ::: "memory");
  asm volatile ("fence rw, rw" ::: "memory");
  /* 元素维度转 tile 维度（向上取整） */
  const size_t TI = div_up(dim_I, (size_t)DIM_I);
  const size_t TJ = div_up(dim_J, (size_t)DIM_J);
  /* 外层块数量（每块包含 tile_I/J 个 tile） */
  const size_t I0 = div_up(TI, tile_I);
  const size_t J0 = div_up(TJ, tile_J);
  for (size_t i0 = 0; i0 < I0; i0++) {
    for (size_t j0 = 0; j0 < J0; j0++) {
      /* 此外层块内实际 tile 数（尾块可能不足） */
      const size_t I = (i0 < I0 - 1) ? tile_I : (TI - (I0 - 1) * tile_I);
      const size_t J = (j0 < J0 - 1) ? tile_J : (TJ - (J0 - 1) * tile_J);
      const acc_t* tile_in1 = (acc_t*)in1 + (i0 * tile_I * DIM_I) * stride_in1 + j0 * tile_J * DIM_J;
      const acc_t* tile_in2 = (acc_t*)in2 + (i0 * tile_I * DIM_I) * stride_in2 + j0 * tile_J * DIM_J;
      acc_t* tile_out = (acc_t*)out + (i0 * tile_I * DIM_I) * stride_out + j0 * tile_J * DIM_J;
      // printf("[INFO][ADD] Outer tile block (%zu, %zu): I=%zu, J=%zu, pre=%p, out=%p\n",
          // i0, j0, I, J, tile_in1, tile_out);
      sp_tiled_add(tile_in1, tile_in2, tile_out, I, J, stride_in1, stride_in2, stride_out);
    }
  }
  asm volatile ("" ::: "memory");
  asm volatile ("fence rw, rw" ::: "memory");
}


// dim_xxx 以“元素”为单位，代表完整矩阵的实际维度
// stride_xxx 以“元素”为单位
static void tiled_add_auto(const size_t dim_I, const size_t dim_J,
    const acc_t* in1, const acc_t* in2, acc_t* out,
    size_t stride_in1, size_t stride_in2, size_t stride_out) {

  const size_t max_tile_ij_in_half_bank_c = ((size_t)sqrt(BANK_C_ROWS * BANK_WIDTH / 2 / (DIM_I * DIM_J) / sizeof(acc_t)));
  const size_t dim_I_padded = (dim_I / DIM_I + (dim_I % DIM_I != 0)) * DIM_I;
  const size_t dim_J_padded = (dim_J / DIM_J + (dim_J % DIM_J != 0)) * DIM_J;
  size_t tile_I = dim_I_padded / DIM_I;
  size_t tile_J = dim_J_padded / DIM_J;
  // printf("max_tile_ij_in_half_bank_c = %d, tile_I = %d, tile_J = %d\n", 
    // (int)max_tile_ij_in_half_bank_c, (int)tile_I, (int)tile_J);
  // tile_I = tile_I < max_tile_ij_in_half_bank_c ? tile_I : max_tile_ij_in_half_bank_c;
  // tile_J = tile_J < max_tile_ij_in_half_bank_c ? tile_J : max_tile_ij_in_half_bank_c;
  tile_I = 1;
  tile_J = 2;

  /* 贪心扩张 */
  while (true) {
    bool increased = false;
    if (is_spad_rows_satisfied(2*tile_I, (tile_J + 2), 0) &&
        (tile_J + 2) * DIM_J <= dim_J_padded) {
      tile_J+=2;
      increased = true;
      // printf("Increase J to %d\n", (int)tile_J);
      continue;
    }
    if (is_spad_rows_satisfied((tile_I + 1), 2*tile_J, 0) &&
        (tile_I + 1) * DIM_I <= dim_I_padded) {
      tile_I++;
      increased = true;
      // printf("Increase I to %d\n", (int)tile_I);
      continue;
    }
    if (!increased) break;
  }
  // printf("Auto tiling selected: I=%d, J=%d\n",
      // (int)tile_I, (int)tile_J);
#if PRINT_TILE
  {
    const int bank_c_rows = 2 * (tile_I * DIM_I * tile_J * DIM_J) * sizeof(acc_t) / BANK_WIDTH;
    printf("tile_I: %d\n", (int)tile_I);
    printf("tile_J: %d\n", (int)tile_J);
    printf("bankC_used_rows: %d\n", bank_c_rows);
    printf("Bank C peak utilization: %f%%\n\n", (float)(bank_c_rows * 100) / BANK_C_ROWS);
  }
  // exit(1);
#endif
  tiled_add(dim_I, dim_J, in1, in2, out, stride_in1, stride_in2, stride_out, tile_I, tile_J);
}

#endif