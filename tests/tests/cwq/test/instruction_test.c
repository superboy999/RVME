#include <stdio.h>
#include <stdalign.h>
#include "../common/inst.h"
#include "../common/matmul.h"

#define M 128
#define N 128
#define K 768
static volatile alignas(64) int8_t A[M][K] = {[0 ... M-1][0 ... K-1] = 1};
static volatile alignas(64) int8_t B[K][N] = {[0 ... K-1][0 ... N-1] = 1};
static volatile alignas(64) int32_t C[M][N] = {[0 ... M-1][0 ... N-1] = 783249};
static volatile alignas(64) int32_t C_T[M][N] = {[0 ... M-1][0 ... N-1] = 1};

void init_matrix() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = (int8_t)(i + j);
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i][j] = (int8_t)(i);
        }
    }
}

void print_matrix() {
    printf("Matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%4d ", A[i][j]);
        }
        printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            printf("%4d ", B[i][j]);
        }
        printf("\n");
    }
}

int main() {
    // mredcmax_w(6, 5); // acc_reg[6] = max(acc_reg[5])
    // mlcte32_spm(TILE_NUM+3, 0, (int)(8 * DIM_J * sizeof(acc_t)), 0); /* C tile -> acc3 */
    // msub_w_mv_i(TILE_NUM+2, TILE_NUM+2, TILE_NUM+3, 0);
    // mscte32_spm(TILE_NUM+2, 0, (int)(8 * DIM_J * sizeof(acc_t)), 0);
    // msrl_w_mm(TILE_NUM+1, TILE_NUM+1, TILE_NUM+0);
    // minv_w_i(TILE_NUM+3, TILE_NUM+3, 0);
    // msync_spm();
    // printf("hello world!\n");
    // msettilem(8);
    // msettilek(32);
    // msettilen(8);
    // // dmaload_spm(0x12000, 0x0, 4);
    // // dmaload_spm((uint64_t)B, 0x8000, 1);
    // // mlae8(0, (uint64_t)B, 32);
    // // dmaload_spm(0x3020, 0x8040, 1);
    // dmaload_spm((uint64_t *)B, 0x8000, 4);
    // dmaload_spm((uint64_t *)A, 0x0, 4);
    // msync_spm();
    // // dmaload_spm(B, 0x8000, 1);
    // // mlbe8_spm(0, 0x8040, 8, 0);
    // mlae8_spm(0, 0x0, 32, 0);
    // msync_spm();
    // // mlbe8_spm(1, 0x8040, 8, 0);

    // ==================== DMA & SPM & Fence Test =====================
    // printf("hello world!\n");
    // init_matrix();
    // print_matrix();
    msettilem(8);
    msettilek(32);
    msettilen(8);
    dmaload_spm((uint64_t *)A, 0x0, 4); // A分布为特殊形式，cachelinesize固定为4 //0
    // dmaload_spm((uint64_t *)A+64*4*2, 0x0, 4); // A分布为特殊形式，cachelinesize固定为4 //0
    // dmaload_spm((uint64_t *)A+64*4*3, 0x0, 4); // A分布为特殊形式，cachelinesize固定为4 //0
    // dmaload_spm((uint64_t *)A+64*4*4, 0x0, 4); // A分布为特殊形式，cachelinesize固定为4 //0
    dmaload_spm((uint64_t *)B, 0x8000, 4); // load B matrix to SPM //1
    dmaload_spm((uint64_t *)C, 0x10000, 4); // load C matrix to SPM //2
    msync_spm(); //3
    // mlae8_spm(0, 0x0, 32, 0); // A的stride固定32 //4
    // mlbe8_spm(1, 0x8000, 32, 0); // B的stride为实际stride（in bytes）//5
    mlce32_spm(4, 0x10000, 32, 0); // C的stride为实际stride（in bytes） //6
    msync_spm(); //7
    // msae8_spm(0, 0x100, 32, 0); //8
    // msbe8_spm(1, 0x8100, 32, 0); //9
    msce32_spm(4, 0x10100, 32, 0); //10
    msync_spm(); //11
    dmastore_spm((uint64_t *)C, 0x10100, 4); // store C matrix back to main memory //12
    msync_spm(); //13
    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    printf("Result Matrix C_T:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%4d ", C[i][j]);
        }
        printf("\n");
    }
    printf("Result Matrix C_T:\n");
    // ===============================================================
    // msettilem(8);
    // msettilek(32);
    // msettilen(8);
    // mlce32(4, (uint64_t *)C, 32);
    // asm volatile ("" ::: "memory");
    // asm volatile ("fence rw, rw" ::: "memory");
    // msce32(4, (uint64_t *)C_T, 32);
    // asm volatile ("" ::: "memory");
    // asm volatile ("fence rw, rw" ::: "memory");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%4d ", C_T[i][j]);
    //     }
    //     printf("\n");
    // }
    return 0;
}