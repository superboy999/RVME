#ifndef __LAYERNORM_V2_H__
#define __LAYERNORM_V2_H__

#include "../common/inst.h"
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

// enum tiled_matmul_type_t {OS, WS, CPU};

static inline size_t div_up(size_t x, size_t y) { return (x + y - 1) / y; }
static inline size_t min_size(size_t a, size_t b) { return a < b ? a : b; }

// static acc_t g_zero_tile[DIM_I * DIM_J] = {0};

//softmax & layernorm 相关常量 tile
static alignas(CACHELINE_SIZE) acc_t shift_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t half_shift_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t dim_J_log2_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t g_zero_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t qa_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t qb_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t qc_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t qs_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const1_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const2_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const3_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const4_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const5_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const6_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const7_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const8_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const9_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const10_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const11_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const12_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const13_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const14_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
static alignas(CACHELINE_SIZE) acc_t const15_tile[DIM_I * DIM_J] = {[0 ... DIM_I*DIM_J-1] = 0};
#define UPDATE_ARRAY(arr, size, val) \
    do { \
        for (size_t _i = 0; _i < (size); _i++) { \
            (arr)[_i] = (val); \
        } \
    } while (0)


static bool is_spad_rows_satisfied(size_t I, size_t J, size_t temp_size) {
  return (((I * DIM_I * J * DIM_J)  * 4 - temp_size) / BANK_WIDTH) <= BANK_C_ROWS;
}

static void sp_tiled_norm(const acc_t* in, acc_t* out,
        size_t I, size_t J, size_t in_row_stride, size_t out_row_stride, int act) {
  
  size_t temp_size; //单位byte
  if (act == SOFTMAX) {
    temp_size = 2 * DIM_I * DIM_J * sizeof(acc_t); // softmax 需要存放两个 tile 大小的临时数据
  } else if (act == LAYERNORM) {
    temp_size = 4 * DIM_I * DIM_J * sizeof(acc_t); // layernorm 需要存放四个 tile 大小的临时数据
  }

  const uint64_t C_sp_addr_start = BANK_A_ROWS * BANK_WIDTH + BANK_B_ROWS * BANK_WIDTH;
  const uint64_t C_temp_sp_addr_start = BANK_A_ROWS * BANK_WIDTH + BANK_B_ROWS * BANK_WIDTH + BANK_C_ROWS * BANK_WIDTH; // 临时存放区，防止覆盖
  
  printf("[INFO] C_sp_addr_start: 0x%lx\n", C_sp_addr_start);
  printf("[INFO] C_temp_sp_addr_start: 0x%lx\n", C_temp_sp_addr_start);
  printf("[INFO] C_sp_addr_end: before 0x%lx\n", BANK_TOTALROWS * BANK_WIDTH);
  
  /* 每次 DMA 最大传输按字节限制换算成 tile 列块数 */
  int C_blocks = (int)(MAX_DMA_BLOCK / (DIM_J * sizeof(acc_t)));
  if (C_blocks < 1) C_blocks = 1;
  if (C_blocks > (int)J) C_blocks = (int)J;
  printf("[INFO] Start sp_tiled_norm(): I=%zu, J=%zu, C_blocks=%d\n", I, J, C_blocks);
  /* Move-in Input ：按 tile 行（J*DIM_J 行）与 tile 列块搬入 */
  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j += (size_t)C_blocks) {
      size_t cols = min_size(J - j, (size_t)C_blocks);
      for (size_t i0 = 0; i0 < DIM_I; i0++) {
        size_t bias_row = i * DIM_I + i0;
        printf("[DEBUG] in:0x%lx\n", (uint64_t)in);
        const acc_t *input_dram_addr = in + bias_row * in_row_stride + j * DIM_J;
        uint64_t input_sp_addr = C_sp_addr_start
            + (uint64_t)((bias_row * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        printf("[DEBUG] input_sp_addr:0x%lx, C_sp_addr_start:0x%lx, + (bias_row:%1x * (J:%1x * DIM_J:%1x) + j:%1x * DIM_J:%1x) * sizeof(acc_t):%1x\n", input_sp_addr, C_sp_addr_start, (unsigned int)bias_row, (unsigned int)J, (unsigned int)DIM_J, (unsigned int)j, (unsigned int)DIM_J, (unsigned int)sizeof(acc_t));
        size_t bytes = cols * DIM_J * sizeof(acc_t);
        printf("[DEBUG] Move-in input: Input[%zu, %zu]-[%zu, %zu] (%zu bytes) from Mem addr 0x%lx-0x%lx to SPM addr 0x%lx-0x%lx\n",
            bias_row, j * DIM_J, bias_row, (j + cols) * DIM_J - 1, bytes, input_dram_addr, input_dram_addr + bytes - 1, input_sp_addr, input_sp_addr + bytes - 1);
        uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
        if (cachelines>=32) cachelines = 0; /* uimm5 */
        dmaload_spm((uint64_t)input_dram_addr, input_sp_addr, 1);
        printf("input_dram_addr = %d, input_sp_addr = %d, cachelines = %d\n", input_dram_addr, input_sp_addr, cachelines);
      }
    }
  }
  msync_spm();
  assert(act == SOFTMAX || act == LAYERNORM); /* 仅支持ReLU、Softmax */
  assert(out != NULL);
  if (act == LAYERNORM) {
  /* ------------------------------------------ ed2 ------------------------------- */
    for( size_t i = 0; i < I; i++) {
      msettilemi(DIM_I);
      // pass 1: get mean
      for (size_t j = 0; j < J; j++) {
        uint64_t input_sp_addr = C_sp_addr_start
          + (uint64_t)(((i * DIM_I) * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        msettileni(DIM_J);
        mlcte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0); /* q -> acc2 Q32.0 */ 
        if (j == 0) {
          /* 载入第一个 tile C，作为初始 sum */
          mmov_mm(TILE_NUM+3, TILE_NUM+2); /* acc3 = C tile */
        } else {
          /* 累加到 sum */
          madd_w_mm(TILE_NUM+3, TILE_NUM+2, TILE_NUM+3); /* acc3 += C tile */
        }
        if (j == J - 1) {
          /* 最后一个 tile 需要进行列缩约，将和缩约到第0行中 */
          mredcadd_w_i(TILE_NUM+3, TILE_NUM+3, 0); /* acc3 col-reduce add */
        }
      }

      /* mean = sum / dim_J */
      // msettileni(DIM_J);
      size_t bytes = DIM_I * DIM_J * sizeof(acc_t);
      uint8_t cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
      if (cachelines>=32) cachelines = 0; /* uimm5 */
      // dmaload_spm((uint64_t)const11_tile, C_temp_sp_addr_start, cachelines);
      // printf("const11_tile = %lld, C_temp_addr_start = %lld\n", (uint64_t)const11_tile, C_temp_sp_addr_start);
      // printf("const11_tile = %lld, C_temp_addr_start = %lld\n", (uint64_t)const11_tile+64*1, C_temp_sp_addr_start+64);
      // printf("const11_tile = %lld, C_temp_addr_start = %lld\n", (uint64_t)const11_tile+64*2, C_temp_sp_addr_start+64*2);
      // printf("const11_tile = %lld, C_temp_addr_start = %lld\n", (uint64_t)const11_tile+64*3, C_temp_sp_addr_start+64*3);
      msync_spm();
      dmaload_spm((uint64_t)const11_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const11_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const11_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const11_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 256 -> acc0 */
      mmul_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+0); /* sum  -> acc3 Q24.8 */
      printf("[DEBUG] Load dim_J_log2\n");
      // dmaload_spm((uint64_t)dim_J_log2_tile, C_temp_sp_addr_start, cachelines);
      msync_spm();
      dmaload_spm((uint64_t)dim_J_log2_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)dim_J_log2_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)dim_J_log2_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)dim_J_log2_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* dim_J_log2 -> acc0 */
      printf("[DEBUG] calculate mean\n");
      msra_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+0); /* mean = sum >> dim_J_log2  Q24.8 算术移位*/

      for (size_t j = 0; j < J; j++) {
        uint64_t input_sp_addr = C_sp_addr_start
          + (uint64_t)(((i * DIM_I) * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        msettileni(DIM_J);
        printf("[DEBUG] load q \n");
        mlcte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0); /* load q -> acc2 */
        // dmaload_spm((uint64_t)const11_tile, C_temp_sp_addr_start, cachelines);
        dmaload_spm((uint64_t)const11_tile, C_temp_sp_addr_start, 1);
        dmaload_spm((uint64_t)const11_tile+64*1, C_temp_sp_addr_start+64, 1);
        dmaload_spm((uint64_t)const11_tile+64*2, C_temp_sp_addr_start+64*2, 1);
        dmaload_spm((uint64_t)const11_tile+64*3, C_temp_sp_addr_start+64*3, 1);
        msync_spm();
        mlce32_spm(TILE_NUM+1, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 256 -> acc1 */
        mmul_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+1); /* q * 256 -> acc2 Q24.8 */
        // msync_spm();
        // dmaload_spm((uint64_t)g_zero_tile, C_temp_sp_addr_start+0, 1);
        // dmaload_spm((uint64_t)g_zero_tile+64, C_temp_sp_addr_start+64, 1);
        // dmaload_spm((uint64_t)g_zero_tile+64*2, C_temp_sp_addr_start+64*2, 1);
        // dmaload_spm((uint64_t)g_zero_tile+64*3, C_temp_sp_addr_start+64*3, 1);
        // msync_spm();
        // mlce32_spm(TILE_NUM+1, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 0 -> acc1 */
        // msub_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+1); /* -mean = 0 - mean */
        msub_w_mv_i(TILE_NUM+2, TILE_NUM+2, TILE_NUM+3, 0); /* q = (q - mean) -> acc2 Q24.8 */

        printf("[DEBUG] store (q - mean) back into spm, addr:%1x\n", input_sp_addr);
        mscte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0); /* acc2 -> C tile (写回 SPM)  SPM 中现为 q - mean Q24.8 */     
      }

      // pass 2: get variance
      for (size_t j = 0; j < J; j++) {
        uint64_t input_sp_addr = C_sp_addr_start
          + (uint64_t)(((i * DIM_I) * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        msettileni(DIM_J);
        mlcte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0); /* load (q - mean) Q24.8 */
        mmov_mm(TILE_NUM+1, TILE_NUM+2);  /* (q - mean) -> acc1 */
        mmul_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+1); /* (q - mean) * (q - mean) -> acc2  Q16.16 */
        // dmaload_spm((uint64_t)const10_tile, C_temp_sp_addr_start, cachelines);
        dmaload_spm((uint64_t)const10_tile, C_temp_sp_addr_start, 1);
        dmaload_spm((uint64_t)const10_tile+64*1, C_temp_sp_addr_start+64, 1);
        dmaload_spm((uint64_t)const10_tile+64*2, C_temp_sp_addr_start+64*2, 1);
        dmaload_spm((uint64_t)const10_tile+64*3, C_temp_sp_addr_start+64*3, 1);
        msync_spm();
        mlce32_spm(TILE_NUM+1, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 8 -> acc1 */
        msra_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+1); /* (q - mean)^2 >> 8  -> acc2 Q24.8 */
        if (j == 0) {
          /* 载入第一个 tile C，作为初始 sum */
          mmov_mm(TILE_NUM+3, TILE_NUM+2); /* acc3 = (q - mean)^2 */
        } else {
          /* 累加到 sum */
          madd_w_mm(TILE_NUM+3, TILE_NUM+2, TILE_NUM+3); /* acc3 += (q - mean)^2 */
        }   
      }
      mredcadd_w_i(TILE_NUM+3, TILE_NUM+3, 0); /* acc3 col-reduce add */
      printf("[DEBUG] calulate variance: Q24.8 \n");
      msra_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+0); /* variance = total_err_sq >> dim_J_log2  Q24.8 算术移位*/
      printf("[DEBUG] store variance back into spm before sqrt:\n");
      printf("[DEBUG] mscte32_spm(%1x)(TILE_NUM+3(%1x), store_addr:(C_temp_sp_addr_start(%1x) + DIM_I(%1x) * DIM_J(%1x) * sizeof(acc_t)(%1x), (int)(J(%1x) * DIM_J(%1x) * sizeof(acc_t))(%1x)), 0)\n", (C_temp_sp_addr_start + DIM_I * DIM_J * sizeof(acc_t)), TILE_NUM+3, C_temp_sp_addr_start, DIM_I, DIM_J, sizeof(acc_t), J, DIM_J, sizeof(acc_t));
      msce32_spm(TILE_NUM+3, C_temp_sp_addr_start + DIM_I * DIM_J * sizeof(acc_t), (int)(DIM_J * sizeof(acc_t)), 0); /* variance -> temp_spm Q24.8 */

      // pass 3: stddev
      msettileni(DIM_J);
      bytes = DIM_I * DIM_J * sizeof(acc_t);
      cachelines = (uint8_t)div_up(bytes, CACHELINE_SIZE);
      if (cachelines>=32) cachelines = 0; /* uimm5 */
      // dmaload_spm((uint64_t)const7_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const7_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const7_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const7_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const7_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 524288 -> acc0 */  
      mmul_w_mm(TILE_NUM+2, TILE_NUM+3, TILE_NUM+0); /* variance * 524288 -> acc2*/
      // dmaload_spm((uint64_t)const3_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const3_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const3_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const3_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const3_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm(); 
      printf("[DEBUG] movein 24\n");
      mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 24 -> acc0*/
      printf("[DEBUG] calculate idx\n");
      msrl_w_mm(TILE_NUM+1, TILE_NUM+2, TILE_NUM+0); /* (variance * 524288) >> 24  : idx  -> acc1 */
      // dmaload_spm((uint64_t)const8_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const8_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const8_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const8_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const8_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 134217728 -> acc0 */  
      mmul_w_mm(TILE_NUM+2, TILE_NUM+3, TILE_NUM+0); /* variance * 134217728 -> acc2*/
      // dmaload_spm((uint64_t)const9_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const9_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const9_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const9_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const9_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /*  27 -> acc0 */
      printf("[DEBUG] calculate frac\n");
      msrl_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+0); /* (variance * 134217728) >> 27 : frac -> acc2*/
      // dmaload_spm((uint64_t)const12_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const12_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const12_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const12_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const12_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 8192 -> acc0 */
      mmax_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+0); /* max(variance, 8192) -> acc3 */
      printf("[DEBUG] calculate A\n");
      msub_w_mm(TILE_NUM+3, TILE_NUM+0, TILE_NUM+3); /* A: {0/negative} -> acc3 */
      // dmaload_spm((uint64_t)const15_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const15_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const15_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const15_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const15_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 31 -> acc0 */
      printf("[DEBUG] calculate B = A >> 31 \n");
      msrl_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+0); /* B = A >> 31 -> acc3 Q32.0 */
      // dmaload_spm((uint64_t)const13_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const13_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const13_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const13_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const13_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm(); 
      mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 65536 (1 Q16.16) -> acc0 */
      mmul_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+0); /* B * 65536 -> acc3 Q16.16 */
      printf("[DEBUG] store idx and frac into spm:\n");
      msce32_spm(TILE_NUM+1, C_temp_sp_addr_start + 2 * DIM_I * DIM_J * sizeof(acc_t), (int)(DIM_J * sizeof(acc_t)), 0); /* idx -> temp */
      msce32_spm(TILE_NUM+2, C_temp_sp_addr_start + 3 * DIM_I * DIM_J * sizeof(acc_t), (int)(DIM_J * sizeof(acc_t)), 0); /* frac -> temp */
      printf("[DEBUG] calculate C\n");
      msub_w_mm(TILE_NUM+1, TILE_NUM+0, TILE_NUM+3); /* C = 1 - B -> acc1 Q16.16 */ 

      // dmaload_spm((uint64_t)const4_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const4_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const4_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const4_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const4_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 1 (1/256 Q24.8) -> acc0 */
      printf("[DEBUG] calculate D\n");
      mmul_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+0); /* D = B * 1/256 -> acc3 Q8.24*/

      // dmaload_spm((uint64_t)const6_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const6_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const6_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const6_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const6_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+2, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /*  16 -> acc2 */
      msrl_w_mm(TILE_NUM+1, TILE_NUM+1, TILE_NUM+2); /* C >> 16 -> acc1 Q32.0*/
      printf("[DEBUG] load idx back \n");
      mlce32_spm(TILE_NUM+2, C_temp_sp_addr_start + 2 * DIM_I * DIM_J * sizeof(acc_t), DIM_J * sizeof(acc_t), 0); /* idx -> acc2 */
      printf("[DEBUG] calculate idx+1\n");
      madd_w_mm(TILE_NUM+0, TILE_NUM+0, TILE_NUM+2); /* ( idx + 1 ) -> acc0 */
      printf("[DEBUG] mlut_w_i for 1/sqrt(idx + 1) and 1/sqrt(idx):\n");
      mlut_w_i(TILE_NUM+0, TILE_NUM+0, 0, 1); /* (idx + 1) -> 1/sqrt(idx + 1) q8.24*/
      mlut_w_i(TILE_NUM+2, TILE_NUM+2, 0, 1); /* idx -> 1/sqrt(idx) q8.24*/
      mmul_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+1); /* C * LUT[idx] Q8.24 */
      mmul_w_mm(TILE_NUM+0, TILE_NUM+0, TILE_NUM+1); /* C * LUT[idx + 1] Q8.24 */
      madd_w_mm(TILE_NUM+0, TILE_NUM+0, TILE_NUM+3); /* (C * LUT[idx + 1] + D) Q8.24 -> acc0 */
      madd_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+3); /* (C * LUT[idx] + D) Q8.24 -> acc2 */
      printf("[DEBUG] calculate diff and frac\n");
      msub_w_mm(TILE_NUM+0, TILE_NUM+0, TILE_NUM+2); /* diff = 1/sqrt(idx + 1) - 1/sqrt(idx) Q8.24*/
      mlce32_spm(TILE_NUM+1, C_temp_sp_addr_start + 3 * DIM_I * DIM_J * sizeof(acc_t), DIM_J * sizeof(acc_t), 0); /* frac -> acc1 */
      // dmaload_spm((uint64_t)const14_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const14_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const14_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const14_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const14_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+3, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /*  5 -> acc3 */
      msra_w_mm(TILE_NUM+0, TILE_NUM+0, TILE_NUM+3); /* diff >> 5 -> acc0 Q13.19 算术右移*/
      mmul_w_mm(TILE_NUM+0, TILE_NUM+0, TILE_NUM+1); /* diff * frac Q8.24*/
      printf("[DEBUG] calculate y0\n");
      madd_w_mm(TILE_NUM+0, TILE_NUM+0, TILE_NUM+2); /* (LUT[idx] + diff * frac) : y0  Q8.24 -> acc0 */

      // dmaload_spm((uint64_t)const6_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const6_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const6_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const6_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const6_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+3, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /*  16 -> acc3 */      
      msrl_w_mm(TILE_NUM+0, TILE_NUM+0, TILE_NUM+3); /* y0 >> 16 Q24.8 -> acc0*/
      mmul_w_mm(TILE_NUM+2, TILE_NUM+0, TILE_NUM+0); /* y0 * y0 -> acc2  Q16.16*/
      // dmaload_spm((uint64_t)const10_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const10_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const10_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const10_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const10_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+3, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /*  8 -> acc3 */
      msrl_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+3); /* y0^2 >> 8 Q24.8 -> acc2 */
      mlce32_spm(TILE_NUM+1, C_temp_sp_addr_start + DIM_I * DIM_J * sizeof(acc_t), DIM_J * sizeof(acc_t), 0); /* x -> acc1 Q24.8*/
      printf("[DEBUG] y0^2 * x\n");
      mmul_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+1); /* y0^2 * x  -> acc2  Q16.16*/
      // dmaload_spm((uint64_t)const5_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const5_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const5_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const5_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const5_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+1, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 1.5(Q16.16) -> acc1 */
      // dmaload_spm((uint64_t)const4_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const4_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const4_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const4_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const4_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+3, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 1 -> acc3 */
      msra_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+3); /* 0.5(x*y^2) -> acc2 Q20.12 算术右移*/
      msub_w_mm(TILE_NUM+1, TILE_NUM+1, TILE_NUM+2); /* (1.5 - 0.5 * y0^2 * x) -> acc1 Q16.16 */
      printf("[DEBUG] calculate y1\n");     
      mmul_w_mm(TILE_NUM+0, TILE_NUM+1, TILE_NUM+0); /* y1 = y0 * (1.5 - 0.5 * y0^2 * x) q8.24 -> acc3 */      

      // dmaload_spm((uint64_t)const6_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const6_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const6_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const6_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const6_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+3, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /*  16 -> acc3 */      
      msrl_w_mm(TILE_NUM+0, TILE_NUM+0, TILE_NUM+3); /* y1 >> 16 Q24.8 -> acc0*/
      mmul_w_mm(TILE_NUM+2, TILE_NUM+0, TILE_NUM+0); /* y1 * y1 -> acc2  Q16.16*/
      // dmaload_spm((uint64_t)const10_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const10_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const10_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const10_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const10_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+3, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /*  8 -> acc3 */
      msrl_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+3); /* y1^2 >> 8 Q24.8 -> acc2 */
      mlce32_spm(TILE_NUM+1, C_temp_sp_addr_start + DIM_I * DIM_J * sizeof(acc_t), DIM_J * sizeof(acc_t), 0); /* x -> acc1 Q16.8*/
      printf("[DEBUG] y1^2 * x\n");
      mmul_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+1); /* y1^2 * x  -> acc2  Q16.16*/
      // dmaload_spm((uint64_t)const5_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const5_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const5_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const5_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const5_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+1, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 1.5(Q16.16) -> acc1 */
      msync_spm();
      // dmaload_spm((uint64_t)const4_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const4_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const4_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const4_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const4_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+3, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /* 1 -> acc3 */
      msra_w_mm(TILE_NUM+2, TILE_NUM+2, TILE_NUM+3); /* 0.5(x*y^2) -> acc2 Q16.16 算术右移*/
      msub_w_mm(TILE_NUM+1, TILE_NUM+1, TILE_NUM+2); /* (1.5 - 0.5 * y1^2 * x) -> acc1 Q16.16 */
      printf("[DEBUG] calculate y2\n");     
      mmul_w_mm(TILE_NUM+3, TILE_NUM+1, TILE_NUM+0); /* y2 = y1 * (1.5 - 0.5 * y1^2 * x) q8.24 -> acc3 */ 
      // dmaload_spm((uint64_t)const10_tile, C_temp_sp_addr_start, cachelines);
      dmaload_spm((uint64_t)const10_tile, C_temp_sp_addr_start, 1);
      dmaload_spm((uint64_t)const10_tile+64*1, C_temp_sp_addr_start+64, 1);
      dmaload_spm((uint64_t)const10_tile+64*2, C_temp_sp_addr_start+64*2, 1);
      dmaload_spm((uint64_t)const10_tile+64*3, C_temp_sp_addr_start+64*3, 1);
      msync_spm();
      mlce32_spm(TILE_NUM+0, C_temp_sp_addr_start, DIM_J * sizeof(acc_t), 0); /*  8 -> acc0 */      
      msrl_w_mm(TILE_NUM+3, TILE_NUM+3, TILE_NUM+0); /* y2 >> 8 Q16.16 -> acc3 */
      printf("[DEBUG] pass3 over\n");

      // pass 4: normalize
      for (size_t j = 0; j < J; j++) {
        uint64_t input_sp_addr = C_sp_addr_start
          + (uint64_t)(((i * DIM_I) * (J * DIM_J) + j * DIM_J) * sizeof(acc_t));
        msettileni(DIM_J);
        mlcte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0); /* load (q - mean) Q24.8*/
        printf("[DEBUG] calculate normalized:\n");       
        mmul_w_mv_i(TILE_NUM+2, TILE_NUM+2, TILE_NUM+3, 0); /* normalized = (q - mean) / stddev Q8.24 -> acc2 */
        mscte32_spm(TILE_NUM+2, input_sp_addr, (int)(J * DIM_J * sizeof(acc_t)), 0); /* acc2 -> C tile (写回 SPM)  SPM 中现为 normalized */
      }
    }
  }
  msync_spm();
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
        dmastore_spm((uint64_t)out_dram_addr, C_sp_addr, cachelines);
      }
    }
  }
/* --------------------------------------------------------------------------- */
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
  if (bank_c_rows > BANK_C_ROWS) { printf("Bank C over-utilized\n"); exit(1); 
    }
  if (tile_I > 65535 || tile_J > 65535) {
    printf("I/J tiling factors must be < 65535\n"); exit(1);
  }
#endif
  if (act == SOFTMAX) {
    const float a = 0.356650;
    const float b = 0.962866;
    const float c = 0.997934;
    const float intermediate_scale = 1 << intermediate_q_bits;
    const acc_t half_intermediate_q_bits = intermediate_q_bits / 2;
    acc_t qb = (acc_t)((b / (2 * a)) * intermediate_scale);
    acc_t qc = (acc_t)((c / a - (b * b) / (4 * a * a)) * intermediate_scale);
    acc_t qln2 = (acc_t) (0.693147 * intermediate_scale);
    acc_t qsln2_inv = (acc_t) (-1.442695 * input_scale_factor * intermediate_scale);
    acc_t qs = (acc_t) (input_scale_factor * intermediate_scale);
    UPDATE_ARRAY(qb_tile, DIM_I * DIM_J, qb);
    UPDATE_ARRAY(qc_tile, DIM_I * DIM_J, qc);
    UPDATE_ARRAY(const1_tile, DIM_I * DIM_J, qln2);
    UPDATE_ARRAY(const2_tile, DIM_I * DIM_J, qsln2_inv);
    UPDATE_ARRAY(shift_tile, DIM_I * DIM_J, intermediate_q_bits);
    UPDATE_ARRAY(half_shift_tile, DIM_I * DIM_J, half_intermediate_q_bits);
    UPDATE_ARRAY(qs_tile, DIM_I * DIM_J, qs);
  } else if (act == LAYERNORM) {
    const float quakeconst1 = 64.0;
    const float quakeconst2 = 24.0;
    const float quakeconst3 = 1.0;
    const float quakeconst4 = 98304.0; //1.5 Q16.16
    const float quakeconst5 = 16.0;
    const float quakeconst6 = 524288.0;  //2^19
    const float quakeconst7 = 134217728.0;  //2^27
    const float quakeconst8 = 27.0;
    const float quakeconst9 = 8.0;
    const float quakecosnt10 = 256.0;
    const float quakeconst11 = 8192.0; //2^13
    const float quakeconst12 = 65536.0; //2^16
    const float quakeconst13 = 5.0;
    const float quakeconst14 = 31.0;
    const float intermediate_scale = 1 << intermediate_q_bits;
    const float dim_J_log2 = log2(dim_J);
    const acc_t half_intermediate_q_bits = intermediate_q_bits / 2;
    acc_t qs = (acc_t) (input_scale_factor * intermediate_scale);
    acc_t qdim_J = (acc_t)(dim_J * (1 << 16));   
    //acc_t qdim_J = (acc_t)(dim_J * intermediate_scale);
    acc_t qquakeconst1 = (acc_t)(quakeconst1 * intermediate_scale);
    acc_t qquakeconst2 = (acc_t)(quakeconst2 * intermediate_scale);
    acc_t qquakeconst3 = (acc_t)(quakeconst3 * intermediate_scale);
    acc_t qquakeconst4 = (acc_t)(quakeconst4 * intermediate_scale);
    acc_t qquakeconst5 = (acc_t)(quakeconst5 * intermediate_scale);
    acc_t qquakeconst6 = (acc_t)(quakeconst6 * intermediate_scale);
    acc_t qquakeconst7 = (acc_t)(quakeconst7 * intermediate_scale);
    acc_t qquakeconst8 = (acc_t)(quakeconst8 * intermediate_scale);
    acc_t qquakeconst9 = (acc_t)(quakeconst9 * intermediate_scale);
    acc_t qquakeconst10 = (acc_t)(quakecosnt10 * intermediate_scale);
    acc_t qquakeconst11 = (acc_t)(quakeconst11 * intermediate_scale);
    acc_t qquakeconst12 = (acc_t)(quakeconst12 * intermediate_scale);
    acc_t qquakeconst13 = (acc_t)(quakeconst13 * intermediate_scale);
    acc_t qquakeconst14 = (acc_t)(quakeconst14 * intermediate_scale);
    UPDATE_ARRAY(shift_tile, DIM_I * DIM_J, intermediate_q_bits);
    UPDATE_ARRAY(half_shift_tile, DIM_I * DIM_J, half_intermediate_q_bits);
    // UPDATE_ARRAY(qs_tile, DIM_I * DIM_J, qs);
    UPDATE_ARRAY(const1_tile, DIM_I * DIM_J, qdim_J);
    UPDATE_ARRAY(const2_tile, DIM_I * DIM_J, qquakeconst1);
    UPDATE_ARRAY(const3_tile, DIM_I * DIM_J, qquakeconst2);
    UPDATE_ARRAY(const4_tile, DIM_I * DIM_J, qquakeconst3); 
    UPDATE_ARRAY(const5_tile, DIM_I * DIM_J, qquakeconst4);
    UPDATE_ARRAY(const6_tile, DIM_I * DIM_J, qquakeconst5);
    UPDATE_ARRAY(const7_tile, DIM_I * DIM_J, qquakeconst6);
    UPDATE_ARRAY(const8_tile, DIM_I * DIM_J, qquakeconst7);
    UPDATE_ARRAY(const9_tile, DIM_I * DIM_J, qquakeconst8);
    UPDATE_ARRAY(const10_tile, DIM_I * DIM_J, qquakeconst9);
    UPDATE_ARRAY(const11_tile, DIM_I * DIM_J, qquakeconst10);
    UPDATE_ARRAY(const12_tile, DIM_I * DIM_J, qquakeconst11);
    UPDATE_ARRAY(const13_tile, DIM_I * DIM_J, qquakeconst12);
    UPDATE_ARRAY(const14_tile, DIM_I * DIM_J, qquakeconst13);
    UPDATE_ARRAY(const15_tile, DIM_I * DIM_J, qquakeconst14);
    UPDATE_ARRAY(dim_J_log2_tile, DIM_I * DIM_J, (acc_t)(dim_J_log2));
  }
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
  asm volatile ("" ::: "memory");
  asm volatile ("fence rw, rw" ::: "memory");
}


// dim_xxx 以“元素”为单位，代表完整矩阵的实际维度
// stride_xxx 以“元素”为单位
static void tiled_norm_auto(const size_t dim_I, const size_t dim_J,
    const acc_t* in, acc_t* out, double in_scale_factor, const size_t intermediate_q_bits,
    size_t stride_in, size_t stride_out, int act) {

  size_t temp_size; //单位byte
  if (act == SOFTMAX) {
    temp_size = 2 * DIM_I * DIM_J * sizeof(acc_t); // softmax 需要存放两个 tile 大小的临时数据
  } else if (act == LAYERNORM) {
    temp_size = 2 * DIM_I * DIM_J * sizeof(acc_t); // layernorm 需要存放两个 tile 大小的临时数据
  }

  const size_t max_tile_ij_in_bank_c = ((size_t)sqrt((BANK_C_ROWS * BANK_WIDTH - temp_size) / (DIM_I * DIM_J) / sizeof(acc_t)));
  const size_t dim_I_padded = (dim_I / DIM_I + (dim_I % DIM_I != 0)) * DIM_I;
  const size_t dim_J_padded = (dim_J / DIM_J + (dim_J % DIM_J != 0)) * DIM_J;
  size_t tile_I = dim_I_padded / DIM_I;
  size_t tile_J = dim_J_padded / DIM_J;
  printf("max_tile_ij_in_bank_c = %d, tile_I = %d, tile_J = %d\n", 
    (int)max_tile_ij_in_bank_c, (int)tile_I, (int)tile_J);
  if (act == LAYERNORM || act == SOFTMAX) {
    tile_I = 1;
  } else {
    tile_I = 1;
    tile_J = 1;
  }
  /* 贪心扩张 */
  while (true) {
    bool increased = false;
    if (is_spad_rows_satisfied(tile_I, tile_J + 1, temp_size) &&
        (tile_J + 1) * DIM_J <= dim_J_padded) {
      tile_J++;
      increased = true;
      printf("Increase J to %d\n", (int)tile_J);
      continue;
    }
    if (is_spad_rows_satisfied(tile_I + 1, tile_J, temp_size) &&
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
}

#endif