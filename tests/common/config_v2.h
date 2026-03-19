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
#define BANK_TOTALROWS 8196
#define BANK_A_ROWS 2052
#define BANK_B_ROWS 2048
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

static inline size_t div_up(size_t x, size_t y) { return (x + y - 1) / y; }
static inline size_t min_size(size_t a, size_t b) { return a < b ? a : b; }

static bool is_spad_rows_satisfied(size_t I, size_t J, size_t K) {
  return ((I * DIM_I * K * DIM_K) / BANK_WIDTH) <= BANK_A_ROWS &&
         ((J * DIM_J * K * DIM_K) / BANK_WIDTH) <= BANK_B_ROWS &&
         ((I * DIM_I * J * DIM_J * sizeof(acc_t)) / BANK_WIDTH) <= BANK_C_ROWS;
}

static bool is_spad_rows_satisfied_2d(size_t I, size_t J, size_t temp_size) {
  return (((I * DIM_I * J * DIM_J)  * 4 - temp_size) / BANK_WIDTH) <= BANK_C_ROWS;
}

static bool is_spad_rows_satisfied_wholeMat(size_t I, size_t J, size_t K) {
  return ((I * K) / BANK_WIDTH) <= BANK_A_ROWS &&
         ((J * K) / BANK_WIDTH) <= BANK_B_ROWS &&
         ((I * J * sizeof(acc_t)) / BANK_WIDTH) <= BANK_C_ROWS;
}

#endif /* __CONFIG_H */