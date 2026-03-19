#ifndef __CONFIG_H
#define __CONFIG_H
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#define PRINT_TILE true
#define ASSERTIONS true

#define ACC_NUM 4
#define TILE_NUM 4
#define DIM_I 8
#define DIM_J 8
#define DIM_K 32
// #define BANK_NUM 3
#define BANK_TOTALROWS 10240
#define BANK_A_ROWS 4096
#define BANK_B_ROWS 4096
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
typedef int64_t full_t;
typedef float scale_t;
typedef uint32_t scale_t_bits;
typedef int32_t scale_acc_t;
typedef uint32_t scale_acc_t_bits;
typedef float acc_scale_t;
typedef uint32_t acc_scale_t_bits;
enum tiled_matmul_type_t {OS, WS, CPU};

static const size_t max_temp_size = 4 * DIM_I * DIM_J * sizeof(acc_t); // 默认 layernorm 所需临时空间大小

static inline size_t div_up(size_t x, size_t y) { return (x + y - 1) / y; }
static inline size_t min_size(size_t a, size_t b) { return a < b ? a : b; }

/* Double Buffer 配置 */
#define BANK_A_BUF_ROWS (BANK_A_ROWS / 2)  /* A bank每个buffer的行数 */
#define BANK_B_BUF_ROWS (BANK_B_ROWS / 2)  /* B bank每个buffer的行数 */
#define BANK_C_BUF_ROWS (BANK_C_ROWS / 2)  /* C bank每个buffer的行数 */

/* Double Buffer 起始地址（字节地址） */
#define BANK_A_BUF0_START (0)
#define BANK_A_BUF1_START (BANK_A_BUF_ROWS * BANK_WIDTH)

#define BANK_B_BUF0_START (BANK_A_ROWS * BANK_WIDTH)
#define BANK_B_BUF1_START (BANK_A_ROWS * BANK_WIDTH + BANK_B_BUF_ROWS * BANK_WIDTH)

#define BANK_C_BUF0_START ((BANK_A_ROWS + BANK_B_ROWS) * BANK_WIDTH)
#define BANK_C_BUF1_START ((BANK_A_ROWS + BANK_B_ROWS) * BANK_WIDTH + BANK_C_BUF_ROWS * BANK_WIDTH)

/* Double Buffer 地址获取辅助函数 */
static inline uint64_t get_A_buf_addr(int buffer_id) {
    return buffer_id == 0 ? BANK_A_BUF0_START : BANK_A_BUF1_START;
}

static inline uint64_t get_B_buf_addr(int buffer_id) {
    return buffer_id == 0 ? BANK_B_BUF0_START : BANK_B_BUF1_START;
}

static inline uint64_t get_C_buf_addr(int buffer_id) {
    return buffer_id == 0 ? BANK_C_BUF0_START : BANK_C_BUF1_START;
}

/* 检查函数 - 原始版本（单buffer） */
static bool is_spad_rows_satisfied(size_t I, size_t J, size_t K) {
  return ((I * DIM_I * K * DIM_K) / BANK_WIDTH) <= BANK_A_ROWS &&
         ((J * DIM_J * K * DIM_K) / BANK_WIDTH) <= BANK_B_ROWS &&
         ((I * DIM_I * J * DIM_J * sizeof(acc_t)) / BANK_WIDTH) <= BANK_C_ROWS;
}

/* 检查函数 - 单个double buffer版本 */
static bool is_spad_rows_satisfied_for_one_buffer(size_t I, size_t J, size_t K) {
  return ((I * DIM_I * K * DIM_K) / BANK_WIDTH) <= BANK_A_BUF_ROWS &&
         ((J * DIM_J * K * DIM_K) / BANK_WIDTH) <= BANK_B_BUF_ROWS &&
         ((I * DIM_I * J * DIM_J * sizeof(acc_t)) / BANK_WIDTH) <= BANK_C_BUF_ROWS;
}

/* 检查函数 - double buffer总体容量（v22版本，相当于两个buffer的总和） */
static bool is_spad_rows_satisfied_v22(size_t I, size_t J, size_t K) {
  return (2 * (I * DIM_I * K * DIM_K) / BANK_WIDTH) <= BANK_A_ROWS &&
         (2 * (J * DIM_J * K * DIM_K) / BANK_WIDTH) <= BANK_B_ROWS &&
         (2 * (I * DIM_I * J * DIM_J * sizeof(acc_t)) / BANK_WIDTH) <= BANK_C_ROWS;
}

static bool is_spad_rows_satisfied_2d(size_t I, size_t J, size_t temp_size) {
  return (((I * DIM_I * J * DIM_J)  * 4) / BANK_WIDTH) <= BANK_C_ROWS;
}

static bool is_spad_rows_satisfied_wholeMat(size_t I, size_t J, size_t K) {
  return ((I * K) / BANK_WIDTH) <= BANK_A_ROWS &&
         ((J * K) / BANK_WIDTH) <= BANK_B_ROWS &&
         ((I * J * sizeof(acc_t)) / BANK_WIDTH) <= BANK_C_ROWS;
}

/* SPM地址计算辅助函数 */

/* A矩阵：按DIM块顺序存储，每个DIM块(8×32)占一行(32字节) */
static inline uint64_t get_A_dim_block_addr(int buffer_id, size_t tile_i, size_t tile_k, size_t total_K) {
    /* A在SPM中按行优先存储DIM块：(0,0), (0,1), ..., (0,total_K-1), (1,0), ... */
    size_t dim_block_index = tile_i * total_K + tile_k;
    return get_A_buf_addr(buffer_id) + (dim_block_index * DIM_I * DIM_K * sizeof(elem_t)); /* 每个DIM块32字节，正好一行 */
}

/* B矩阵：按自然顺序存储，每行32字节 */
static inline uint64_t get_B_block_addr(int buffer_id, size_t k0, size_t tile_j, size_t total_J) {
    /* B的每个tile是32×8，需要根据实际布局计算地址 */
    /* 假设按行存储，每行32字节 */
    size_t byte_offset = (k0 * total_J * DIM_J + tile_j * DIM_J) * sizeof(elem_t);
    return get_B_buf_addr(buffer_id) + byte_offset;
}

/* C矩阵：按自然顺序存储，每行32字节 */
static inline uint64_t get_C_block_addr(int buffer_id, size_t i0, size_t j_tile, size_t total_J) {
    /* C的每个tile是8×8，每个元素4字节，共256字节，需要8行(32字节/行) */
    size_t byte_offset = (i0 * total_J * DIM_J + j_tile * DIM_J) * sizeof(acc_t);
    return get_C_buf_addr(buffer_id) + byte_offset;
}


#endif /* __CONFIG_H */