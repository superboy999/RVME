#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>
#include "../common/inst.h"
#define BANK_TOTALROWS 8196
#define BANK_A_ROWS 2052
#define BANK_B_ROWS 2048
#define BANK_C_ROWS (BANK_TOTALROWS - BANK_A_ROWS - BANK_B_ROWS)
#define CACHELINE_SIZE 64
#define M 8
#define N 8
#define K 32
#define BANK_WIDTH 32
#define TILE_NUM 4
#define DIM_I 8
#define DIM_J 8
#define DIM_K 32

typedef int8_t elem_t;
typedef int32_t acc_t;
static alignas(CACHELINE_SIZE) elem_t input[M][K] = {[0 ... M-1][0 ... K-1] = 1};
static alignas(CACHELINE_SIZE) acc_t data_input[M][N] = {[0 ... M-1][0 ... N-1] = 1};

int main()
{
    uint64_t stride = 16 * sizeof(int8_t); //indicate the row size
    uint32_t c = 0;
    msettilemi(M);
    msettileni(N);
    msettileki(K);

    uint64_t A_sp = 0;
    uint64_t C_sp = (BANK_A_ROWS + BANK_B_ROWS) * BANK_WIDTH;

    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    dmaload_spm((uint64_t)input, A_sp, 4);
    dmaload_spm((uint64_t)data_input, C_sp, 1);
    dmaload_spm((uint64_t)data_input+64, C_sp+64, 1);
    dmaload_spm((uint64_t)data_input+64*2, C_sp+64*2, 1);
    dmaload_spm((uint64_t)data_input+64*3, C_sp+64*3, 1);
    msync_spm();

    mlae8_spm(0, A_sp, (int)(DIM_K * sizeof(elem_t)), 0);
    mlce32_spm(TILE_NUM+0, C_sp, DIM_J * sizeof(acc_t), 0); /* 8 -> acc0 */

    madd_w_mm(TILE_NUM+0, TILE_NUM+0, TILE_NUM+0);
    // mmovb_m_x(TILE_NUM+0, 128, 8);
    
    mmovb_m_x(TILE_NUM+0, 128, 8);

    // asm volatile ("" ::: "memory");
    // asm volatile ("fence rw, rw" ::: "memory");
  
    // mmov_mm(TILE_NUM+0, 0);
    c = mmovb_x_m(TILE_NUM+0, 8);
    msce32_spm(TILE_NUM+0, C_sp, DIM_J * sizeof(acc_t), 0);
    msync_spm();
    // dmastore_spm((uint64_t)input, A_sp, 4);
    dmastore_spm((uint64_t)data_input, C_sp, 1);
    dmastore_spm((uint64_t)data_input+64, C_sp+64, 1);
    dmastore_spm((uint64_t)data_input+64*2, C_sp+64*2, 1);
    dmastore_spm((uint64_t)data_input+64*3, C_sp+64*3, 1);
    msce32_spm(TILE_NUM+0, C_sp, DIM_J * sizeof(acc_t), 0);
    msync_spm();
    dmastore_spm((uint64_t)data_input, C_sp, 1);
    dmastore_spm((uint64_t)data_input+64, C_sp+64, 1);
    dmastore_spm((uint64_t)data_input+64*2, C_sp+64*2, 1);
    dmastore_spm((uint64_t)data_input+64*3, C_sp+64*3, 1);
    // dmaload_spm((uint64_t)input, A_sp, 4);
    // dmaload_spm((uint64_t)data_input, C_sp, 1);
    // dmaload_spm((uint64_t)data_input+64, C_sp+64, 1);
    // dmaload_spm((uint64_t)data_input+64*2, C_sp+64*2, 1);
    // dmaload_spm((uint64_t)data_input+64*3, C_sp+64*3, 1);

    // mlae8_spm(0, A_sp, (int)(DIM_K * sizeof(elem_t)), 0);
    // mlce32_spm(TILE_NUM+0, C_sp, DIM_J * sizeof(acc_t), 0); /* 8 -> acc0 */

    // mmovb_m_x(0, 128, 8);


    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            printf("%d ", data_input[i][j]);
        }
        printf("\n");
    }

    msce32_spm(TILE_NUM+0, C_sp, DIM_J * sizeof(acc_t), 0);
    msync_spm();
    dmastore_spm((uint64_t)data_input, C_sp, 1);
    dmastore_spm((uint64_t)data_input+64, C_sp+64, 1);
    dmastore_spm((uint64_t)data_input+64*2, C_sp+64*2, 1);
    dmastore_spm((uint64_t)data_input+64*3, C_sp+64*3, 1);

    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            printf("%d ", data_input[i][j]);
        }
        printf("\n");
    }
    
    printf("mov c data = %d\n", c);
    printf("test finish!!\n");
    return 0;
}