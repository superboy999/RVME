#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__

#include "inst.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdalign.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#define PRINT_TILE true
#define ASSERTIONS true

#define ACC_NUM 4
#define TILE_NUM 4
#define DIM_I 8
#define DIM_J 8
#define DIM_K 32
// #define BANK_NUM 3
#define BANK_TOTALROWS 4096
#define BANK_A_ROWS 1024
#define BANK_B_ROWS 1024
#define BANK_C_ROWS (BANK_TOTALROWS - BANK_A_ROWS - BANK_B_ROWS)
#define BANK_WIDTH 32
#define CACHELINE_SIZE 64
#define MAX_DMA_BLOCK (1 * CACHELINE_SIZE)

#define NO_ACTIVATION 0
#define RELU 1
#define LAYERNORM 2
#define IGELU 3
#define SOFTMAX 4

typedef int8_t elem_t;
static const elem_t elem_t_max = 127;
static const elem_t elem_t_min = -128;
typedef int32_t acc_t;

// typedef float scale_t;
// typedef uint32_t scale_t_bits;
// typedef int32_t scale_acc_t;
// typedef uint32_t scale_acc_t_bits;
// typedef float acc_scale_t;
// typedef uint32_t acc_scale_t_bits;

// enum tiled_matmul_type_t {OS, WS, CPU};

static inline size_t div_up(size_t x, size_t y) { return (x + y - 1) / y; }
static inline size_t min_size(size_t a, size_t b) { return a < b ? a : b; }

static acc_t g_zero_tile[DIM_I * DIM_J] = {0};
static acc_t qa_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 1};
static acc_t qb_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 1};
static acc_t qc_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 1};
static acc_t qs_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 1};
static acc_t const1_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 1};
static acc_t const2_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 1};
static acc_t shift_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 1};
static acc_t half_shift_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 1};
static acc_t lshift_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 9};
static acc_t rshift_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 22};
#define UPDATE_ARRAY(arr, size, val) \
    do { \
        for (size_t _i = 0; _i < (size); _i++) { \
            (arr)[_i] = (val); \
        } \
    } while (0)


static bool is_spad_rows_satisfied(size_t I, size_t J) {
  return ((I * DIM_I * J * DIM_J * sizeof(acc_t)) / BANK_WIDTH) <= BANK_C_ROWS;
}

static void sp_tiled_norm(const acc_t* in, acc_t* out,
        size_t I, size_t J, size_t in_row_stride, size_t out_row_stride, int act) {
  const uint64_t C_sp_addr_start = BANK_A_ROWS * BANK_WIDTH + BANK_B_ROWS * BANK_WIDTH;
  const uint64_t C_temp_sp_addr_start = BANK_A_ROWS * BANK_WIDTH + BANK_B_ROWS * BANK_WIDTH + BANK_C_ROWS * BANK_WIDTH; // 临时存放区，防止覆盖
  /* 每次 DMA 最大传输按字节限制换算成 tile 列块数 */
  int C_blocks = (int)(MAX_DMA_BLOCK / (DIM_J * sizeof(acc_t)));
  if (C_blocks < 1) C_blocks = 1;
  if (C_blocks > (int)J) C_blocks = (int)J;
  printf("[INFO] Start sp_tiled_norm(): I=%zu, J=%zu, C_blocks=%d\n", I, J, C_blocks);
  /* Move-in Input ：按 tile 行（J*DIM_J 行）与 tile 列块搬入 */
  for (size_t i0 = 0; i0 < I * DIM_I; i0++) {
    for (size_t j = 0; j < J; j += (size_t)C_blocks) {
      size_t cols = min_size(J - j, (size_t)C_blocks); 
      printf("[DEBUG] in:0x%lx\n", (uint64_t)in);
      const acc_t *input_dram_addr = in + i0 * in_row_stride + j * DIM_J;
      uint64_t input_sp_addr = C_sp_addr_start
          + (uint64_t)((i0 * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
      printf("[DEBUG] input_sp_addr = 0x%lx = %llu + (%llu * %llu + %llu) * %llu\n", input_sp_addr, C_sp_addr_start, i0, (J * DIM_J), j * DIM_J, sizeof(acc_t));
      size_t bytes = cols * DIM_J * sizeof(acc_t);
      printf("[DEBUG] Move-in input: Input[%zu, %zu]-[%zu, %zu] (%zu bytes) from Mem addr 0x%lx-0x%lx to SPM addr 0x%lx-0x%lx\n",
          i0, j * DIM_J, i0, (j + cols) * DIM_J - 1, bytes, input_dram_addr, input_dram_addr + bytes - 1, input_sp_addr, input_sp_addr + bytes - 1);
      uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
      if (cachelines>=32) cachelines = 0; /* uimm5 */
      // dmaload_spm((uint64_t)input_dram_addr, input_sp_addr, cachelines);
      // dmaload_spm((uint64_t)input_dram_addr, input_sp_addr, 1);
      // dmaload_spm((uint64_t)input_dram_addr+64, input_sp_addr+64, 1);
      // dmaload_spm((uint64_t)input_dram_addr+64*2, input_sp_addr+64*2, 1);
      // dmaload_spm((uint64_t)input_dram_addr+64*3, input_sp_addr+64*3, 1);
      msettilemi(DIM_I);
      msettileni(DIM_J);
      msettileki(DIM_K);
      mlce32(TILE_NUM+0, input_dram_addr, (int)( DIM_J * sizeof(acc_t)));
      mlce32(TILE_NUM+2, input_dram_addr, (int)( DIM_J * sizeof(acc_t)));
      mlce32(TILE_NUM+3, input_dram_addr, (int)( DIM_J * sizeof(acc_t)));
    }
  }
  // msync_spm();
  assert(act == SOFTMAX || act == LAYERNORM); /* 仅支持ReLU、Softmax */
  assert(out != NULL);
  if (act == SOFTMAX) {
    for (size_t i = 0; i < I; i++) {
      msettilemi(DIM_I);
      // pass 1: get max_q
      for (size_t j = 0; j < J; j++) {
        uint64_t input_sp_addr = C_sp_addr_start
          + (uint64_t)(((i * DIM_I) * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        msettileni(DIM_J);
        if (j == 0) {
          /* 载入第一个 tile C，作为初始 max */
          // mlcte32_spm(TILE_NUM+3, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0); /* C tile -> acc3 */
        } else {
          /* 载入下一个 tile C，和当前 max 比较，更新 max */
          // mlcte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0); /* C tile -> acc2 */

          mmax_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+2); /* acc3 >= max(acc3, acc2) */
        }
        if (j == J - 1) {
          /* 最后一个 tile 需要进行列缩约，将最大值缩约到第0行中 */
          // mredcmax_w(TILE_NUM+3, TILE_NUM+3); /* acc3 col-reduce max */
        }
      }
      for (size_t j = 0; j < J; j++) {
        uint64_t input_sp_addr = C_sp_addr_start
          + (uint64_t)(((i * DIM_I) * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        msettileni(DIM_J);
        // mlcte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
        msub_w_mv_i(TILE_NUM+2, TILE_NUM+2, TILE_NUM+3, 0); /* q = q - max */
        // mscte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0); /* acc2 -> C tile (写回 SPM) */
      }
      // pass 2: calculate iexp(q_tilde) and sum(q_tilde)
      for (size_t j = 0; j < J; j++) {
        uint64_t input_sp_addr = C_sp_addr_start
          + (uint64_t)(((i * DIM_I) * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        msettileni(DIM_J);
        size_t bytes = DIM_I * DIM_J * sizeof(acc_t);
        uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
        if (cachelines>=32) cachelines = 0; /* uimm5 */
        // mlcte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
        /* 加载-qln2_inv矩阵 */
        printf("[DEBUG] Load const2_tile\n");
        // dmaload_spm((uint64_t)const2_tile, C_temp_sp_addr_start+0, 1);
        // dmaload_spm((uint64_t)const2_tile+64, C_temp_sp_addr_start+64, 1);
        // dmaload_spm((uint64_t)const2_tile+64*2, C_temp_sp_addr_start+64*2, 1);
        // dmaload_spm((uint64_t)const2_tile+64*3, C_temp_sp_addr_start+64*3, 1);
        // msync_spm();
        // mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0);
        printf("[DEBUG] z = q * -qln2_inv\n");
        mmul_w_mm(TILE_NUM+1, TILE_NUM+2, TILE_NUM+0); /* z = q * -qln2_inv */
        /* 加载shift矩阵 */
        printf("[DEBUG] Load shift_tile\n");
        // dmaload_spm((uint64_t)shift_tile, C_temp_sp_addr_start, cachelines);
        // dmaload_spm((uint64_t)shift_tile, C_temp_sp_addr_start+0, 1);
        // dmaload_spm((uint64_t)shift_tile+64, C_temp_sp_addr_start+64, 1);
        // dmaload_spm((uint64_t)shift_tile+64*2, C_temp_sp_addr_start+64*2, 1);
        // dmaload_spm((uint64_t)shift_tile+64*3, C_temp_sp_addr_start+64*3, 1);
        // msync_spm();
        // mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0);
        printf("[DEBUG] z >> shift\n");
        msrl_w_mm(TILE_NUM+1, TILE_NUM+1, TILE_NUM+0); /* z = z >> 16 */
        /* q = q * qs */
        printf("[DEBUG] Load qs_tile\n");
        // dmaload_spm((uint64_t)qs_tile, C_temp_sp_addr_start, cachelines);
        // dmaload_spm((uint64_t)qs_tile, C_temp_sp_addr_start+0, 1);
        // dmaload_spm((uint64_t)qs_tile+64, C_temp_sp_addr_start+64, 1);
        // dmaload_spm((uint64_t)qs_tile+64*2, C_temp_sp_addr_start+64*2, 1);
        // dmaload_spm((uint64_t)qs_tile+64*3, C_temp_sp_addr_start+64*3, 1);
        
        // msync_spm();
        // mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0);
        printf("[DEBUG] q = q * qs\n");
        mmul_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+0); /* q = q * qs */
        /* 加载qln2矩阵 */
        printf("[DEBUG] Load const1_tile\n");
        // dmaload_spm((uint64_t)const1_tile, C_temp_sp_addr_start, cachelines);
        // dmaload_spm((uint64_t)const1_tile, C_temp_sp_addr_start+0, 1);
        // dmaload_spm((uint64_t)const1_tile+64, C_temp_sp_addr_start+64, 1);
        // dmaload_spm((uint64_t)const1_tile+64*2, C_temp_sp_addr_start+64*2, 1);
        // dmaload_spm((uint64_t)const1_tile+64*3, C_temp_sp_addr_start+64*3, 1);
        // msync_spm();
        // mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0);
        printf("[DEBUG] qp = z * qln2 + q\n");
        mmul_w_mm(TILE_NUM+0, TILE_NUM+1, TILE_NUM+0); /* qp = z * qln2 */
        printf("[DEBUG] qp = qp + q\n");
        madd_w_mm(TILE_NUM+2, TILE_NUM+0, TILE_NUM+2); /* qp = qp + q */
        /* 加载qb矩阵 */
        printf("[DEBUG] Load qb_tile\n");
        // dmaload_spm((uint64_t)qb_tile, C_temp_sp_addr_start, cachelines);
        // dmaload_spm((uint64_t)qb_tile, C_temp_sp_addr_start+0, 1);
        // dmaload_spm((uint64_t)qb_tile+64, C_temp_sp_addr_start+64, 1);
        // dmaload_spm((uint64_t)qb_tile+64*2, C_temp_sp_addr_start+64*2, 1);
        // dmaload_spm((uint64_t)qb_tile+64*3, C_temp_sp_addr_start+64*3, 1);
        // msync_spm();
        // mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0);
        printf("[DEBUG] qp = qp + b\n");
        madd_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+0); /* qp = qp + b */
        /* 加载half_shift矩阵 */
        printf("[DEBUG] Load half_shift_tile\n");
        // dmaload_spm((uint64_t)half_shift_tile, C_temp_sp_addr_start, cachelines);
        // dmaload_spm((uint64_t)half_shift_tile, C_temp_sp_addr_start+0, 1);
        // dmaload_spm((uint64_t)half_shift_tile+64, C_temp_sp_addr_start+64, 1);
        // dmaload_spm((uint64_t)half_shift_tile+64*2, C_temp_sp_addr_start+64*2, 1);
        // dmaload_spm((uint64_t)half_shift_tile+64*3, C_temp_sp_addr_start+64*3, 1);
        // msync_spm();
        // mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0);
        printf("[DEBUG] qp >> (shift/2)\n");
        msrl_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+0); /* qp >> (shift/2) */
        /* 计算 q_exp = qp * qp + qc */
        mmov_mm(TILE_NUM+0, TILE_NUM+2);
        printf("[DEBUG] q_exp = qp * qp\n");
        mmul_w_mm(TILE_NUM+2, TILE_NUM+0, TILE_NUM+2); /* q_exp = qp * qp */
        /* 加载qc矩阵 */
        printf("[DEBUG] Load qc_tile\n");
        // dmaload_spm((uint64_t)qc_tile, C_temp_sp_addr_start, cachelines);
        // dmaload_spm((uint64_t)qc_tile, C_temp_sp_addr_start+0, 1);
        // dmaload_spm((uint64_t)qc_tile+64, C_temp_sp_addr_start+64, 1);
        // dmaload_spm((uint64_t)qc_tile+64*2, C_temp_sp_addr_start+64*2, 1);
        // dmaload_spm((uint64_t)qc_tile+64*3, C_temp_sp_addr_start+64*3, 1);
        // msync_spm();
        // mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0);
        printf("[DEBUG] q_exp = q_exp + qc\n");
        madd_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+0); /* q_exp = q_exp + qc */
        printf("[DEBUG] q_exp >> z\n");
        msrl_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+1); /* q_exp >> z */
        if (j == 0) {
          /* acc0 作为 sum_exp 的初值 */
          mmov_mm(TILE_NUM+3, TILE_NUM+2); /* acc3 = q_exp */
        } else {
          /* 累加到 sum_exp */
          madd_w_mm(TILE_NUM+3, TILE_NUM+2, TILE_NUM+3); /* acc0 += q_exp */
        }
        // mscte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0); /* acc1 -> C tile (写回 SPM) */   
        msce32(TILE_NUM+2, in, (int)(J * DIM_J * sizeof(acc_t)));  
      }
      printf("[DEBUG] sum_exp in acc3 before col-reduce add:\n");
      // mredcadd_w(TILE_NUM+3, TILE_NUM+3); /* acc3 col-reduce add */
      // dmaload_spm((uint64_t)lshift_tile, C_temp_sp_addr_start, 4);
      // dmaload_spm((uint64_t)lshift_tile, C_temp_sp_addr_start+0, 1);
      // dmaload_spm((uint64_t)lshift_tile+64, C_temp_sp_addr_start+64, 1);
      // dmaload_spm((uint64_t)lshift_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      // dmaload_spm((uint64_t)lshift_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      // msync_spm();
      // mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0);
      msll_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+0);
      // dmaload_spm((uint64_t)rshift_tile, C_temp_sp_addr_start, 4);
      // dmaload_spm((uint64_t)rshift_tile, C_temp_sp_addr_start+0, 1);
      // dmaload_spm((uint64_t)rshift_tile+64, C_temp_sp_addr_start+64, 1);
      // dmaload_spm((uint64_t)rshift_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      // dmaload_spm((uint64_t)rshift_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      // msync_spm();
      // mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0);
      msll_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+0);
      // mlut_w_i(TILE_NUM+3, TILE_NUM+3, 0, 0); /* sum_exp -> 1/sum_exp */
      // pass 3: normalize
      for (size_t j = 0; j < J; j++) {
        uint64_t input_sp_addr = C_sp_addr_start
          + (uint64_t)(((i * DIM_I) * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        msettileni(DIM_J);
        size_t bytes = DIM_I * DIM_J * sizeof(acc_t);
        uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
        if (cachelines>=32) cachelines = 0;
        // mlcte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0);
        printf("[DEBUG] Load half_shift_tile\n");
        // dmaload_spm((uint64_t)half_shift_tile, C_temp_sp_addr_start, cachelines);
        // dmaload_spm((uint64_t)half_shift_tile, C_temp_sp_addr_start+0, 1);
        // dmaload_spm((uint64_t)half_shift_tile+64, C_temp_sp_addr_start+64, 1);
        // dmaload_spm((uint64_t)half_shift_tile+64*2, C_temp_sp_addr_start+64*2, 1);
        // dmaload_spm((uint64_t)half_shift_tile+64*3, C_temp_sp_addr_start+64*3, 1);
        // msync_spm();
        // mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0);
        printf("[DEBUG] q >> (shift/2)\n");
        msrl_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+0); /* q >> (shift/2) */
        printf("[DEBUG] q * (1/sum_exp)\n");
        mmul_w_mv_i(TILE_NUM+2, TILE_NUM+2, TILE_NUM+3, 0); /* q * (1/sum_exp) */
        msce32(TILE_NUM+2, in, (int)(J * DIM_J * sizeof(acc_t)));
        // mscte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0); /* acc2 -> C tile (写回 SPM) */
      }
    }
  } else if (act == LAYERNORM) {
    /* TODO */
  }
  // msync_spm();
  /* Move-out Output ：按 tile 行（J*DIM_J 行）与 tile 列块搬出 */
  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j += (size_t)C_blocks) {
      size_t cols = min_size(J - j, (size_t)C_blocks);
      printf("[DEBUG] cols = min(%zu, %zu) = %zu\n", J - j, (size_t)C_blocks, cols);
      for (size_t i0 = 0; i0 < DIM_I; i0++) {
        size_t row = i * DIM_I + i0;
        acc_t *out_dram_addr = out + row * out_row_stride + j * DIM_J;
        uint64_t C_sp_addr = C_sp_addr_start
            + (uint64_t)((row * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        size_t bytes = cols * DIM_J * sizeof(acc_t);
        printf("[DEBUG] Move-out C[%zu, %zu]-[%zu, %zu] (%zu bytes) from SPM addr 0x%lx-0x%lx to Mem addr 0x%lx-0x%lx\n",
            row, j * DIM_J, row, (j + cols) * DIM_J - 1, bytes, C_sp_addr, C_sp_addr + bytes - 1, out_dram_addr, out_dram_addr + bytes - 1);
        uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
        if (cachelines>=32) cachelines = 0; /* uimm5 */
        // dmastore_spm((uint64_t)out_dram_addr, C_sp_addr, cachelines);
      }
    }
  }
}


static void tiled_norm(size_t dim_I, size_t dim_J, const acc_t* in, acc_t* out,
    double input_scale_factor, const acc_t intermediate_q_bits,
    size_t stride_in, size_t stride_out, size_t tile_I, size_t tile_J, int act) {
#if ASSERTIONS
  if (tile_I == 0 || tile_J == 0) {
    printf("tile factors must be positive\n");
    exit(1);
  }
  const size_t dim_I_padded = (dim_I / DIM_I + (dim_I % DIM_I != 0)) * DIM_I;
  const size_t dim_J_padded = (dim_J / DIM_J + (dim_J % DIM_J != 0)) * DIM_J;
  if (tile_I * DIM_I > dim_I_padded) {
    printf("tile_I too large\n"); exit(1);
  }
  if (tile_J * DIM_J > dim_J_padded) {
    printf("tile_J too large\n"); exit(1);
  }
  const int bank_c_rows = (tile_I * DIM_I * tile_J * DIM_J) / BANK_WIDTH;
  if (bank_c_rows > BANK_C_ROWS) { printf("Bank C over-utilized\n"); exit(1); }
  if (tile_I > 65535 || tile_J > 65535) {
    printf("I/J tiling factors must be < 65535\n"); exit(1);
  }
#endif
  // if (act == SOFTMAX) {
  //   const float a = 0.356650;
  //   const float b = 0.962866;
  //   const float c = 0.997934;
  //   const float intermediate_scale = 1 << intermediate_q_bits;
  //   const acc_t half_intermediate_q_bits = intermediate_q_bits / 2;
  //   acc_t qb = (acc_t)((b / (2 * a)) * intermediate_scale);
  //   acc_t qc = (acc_t)((c / a - (b * b) / (4 * a * a)) * intermediate_scale);
  //   acc_t qln2 = (acc_t) (0.693147 * intermediate_scale);
  //   acc_t qsln2_inv = (acc_t) (-1.442695 * input_scale_factor * intermediate_scale);
  //   acc_t qs = (acc_t) (input_scale_factor * intermediate_scale);
  //   UPDATE_ARRAY(qb_tile, DIM_I * DIM_J, qb);
  //   UPDATE_ARRAY(qc_tile, DIM_I * DIM_J, qc);
  //   UPDATE_ARRAY(const1_tile, DIM_I * DIM_J, qln2);
  //   UPDATE_ARRAY(const2_tile, DIM_I * DIM_J, qsln2_inv);
  //   UPDATE_ARRAY(shift_tile, DIM_I * DIM_J, intermediate_q_bits);
  //   UPDATE_ARRAY(half_shift_tile, DIM_I * DIM_J, half_intermediate_q_bits);
  //   UPDATE_ARRAY(qs_tile, DIM_I * DIM_J, qs);
  // }
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
      const acc_t* tile_in = (acc_t*)in + (i0 * tile_I * DIM_I) * stride_in + j0 * tile_J * DIM_J;
      acc_t* tile_out = NULL;
      printf("[DEBUG] j0 = %zu, J0 - 1 = %zu\n", j0, J0 - 1);
      if (j0 == J0 - 1) {
        tile_out = (acc_t*)out + (i0 * tile_I * DIM_I) * stride_out + j0 * tile_J * DIM_J;
      }
      printf("[INFO] Outer tile block (%zu, %zu): I=%zu, J=%zu, pre=%p, out=%p\n",
          i0, j0, I, J, tile_in, tile_out);
      if ((act == LAYERNORM || act == SOFTMAX)) {
        sp_tiled_norm(tile_in, tile_out, I, J, stride_in, stride_out, act);
      } else {
        // sp_tiled_norm_cpu(tile_in, out, I, J, stride_in, stride_out, act);
      }
    }
  }
}


// dim_xxx 以“元素”为单位，代表完整矩阵的实际维度
// stride_xxx 以“元素”为单位
static void tiled_norm_auto(const size_t dim_I, const size_t dim_J,
    const acc_t* in, acc_t* out, double in_scale_factor, const size_t intermediate_q_bits,
    size_t stride_in, size_t stride_out, int act) {
  asm volatile ("" ::: "memory");
  asm volatile ("fence rw, rw" ::: "memory");

  const size_t max_tile_ij_in_bank_c = ((size_t)sqrt(BANK_C_ROWS * BANK_WIDTH / (DIM_I * DIM_J) / sizeof(acc_t)));
  const size_t dim_I_padded = (dim_I / DIM_I + (dim_I % DIM_I != 0)) * DIM_I;
  const size_t dim_J_padded = (dim_J / DIM_J + (dim_J % DIM_J != 0)) * DIM_J;
  size_t tile_I = dim_I_padded / DIM_I;
  size_t tile_J = dim_J_padded / DIM_J;
  printf("max_tile_ij_in_bank_c = %d, tile_I = %d, tile_J = %d\n", 
    (int)max_tile_ij_in_bank_c, (int)tile_I, (int)tile_J);
  size_t temp_size = 0;
  if (act == SOFTMAX) {
    temp_size = 2 * DIM_I * DIM_J * sizeof(acc_t); // 用于临时存放 DMA 加载的数据 in bytes
  } else if (act == LAYERNORM) {
    temp_size = 4 * DIM_I * DIM_J * sizeof(acc_t); // 用于临时存放 DMA 加载的数据 in bytes
  }
  if (act == LAYERNORM || act == SOFTMAX) {
    tile_I = 1;
  } else {
    tile_I = 1;
    tile_J = 1;
  }
  /* 贪心扩张 */
  while (true) {
    bool increased = false;
    if (is_spad_rows_satisfied(tile_I, tile_J + 1) &&
        (tile_J + 1) * DIM_J <= dim_J_padded) {
      tile_J++;
      increased = true;
      printf("Increase J to %d\n", (int)tile_J);
      continue;
    }
    if (is_spad_rows_satisfied(tile_I + 1, tile_J) &&
        (tile_I + 1) * DIM_I <= dim_I_padded) {
      tile_I++;
      increased = true;
      printf("Increase I to %d\n", (int)tile_I);
      continue;
    }
    if (!increased) break;
  }
  printf("Auto tiling selected: I=%d, J=%d, act=%s\n",
      (int)tile_I, (int)tile_J,
      act == LAYERNORM ? "LAYERNORM" : (act == SOFTMAX ? "SOFTMAX" : "NONE"));
#if PRINT_TILE
  {
    const int bank_c_rows = (tile_I * DIM_I * tile_J * DIM_J) * sizeof(acc_t) / BANK_WIDTH;
    printf("tile_I: %d\n", (int)tile_I);
    printf("tile_J: %d\n", (int)tile_J);
    printf("bankC_used_rows: %d\n", bank_c_rows);
    printf("Bank C utilization: %f%%\n\n", (float)(bank_c_rows * 100) / BANK_C_ROWS);
  }
  // exit(1);
#endif
  tiled_norm(dim_I, dim_J, in, out, in_scale_factor, intermediate_q_bits, stride_in, stride_out, tile_I, tile_J, act);

  asm volatile ("" ::: "memory");
  asm volatile ("fence rw, rw" ::: "memory");
}

#endif