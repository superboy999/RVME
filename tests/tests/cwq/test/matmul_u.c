#include "../common/inst.h"
#include "../common/matmul_u.h"
#include <stdio.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

// #define M 128
// #define N 128
// #define K 768
#define M 64
#define N 64
#define K 64
static alignas(64) uint8_t A[M][K] = {[0 ... M-1][0 ... K-1] = 1};
static alignas(64) uint8_t AR[M][K] = {[0 ... M-1][0 ... K-1] = 1};
static alignas(64) uint8_t B[K][N] = {[0 ... K-1][0 ... N-1] = 1};
static alignas(64) uint32_t C[M][N] = {[0 ... M-1][0 ... N-1] = 1};
static alignas(64) uint32_t C_G[M][N] = {[0 ... M-1][0 ... N-1] = 0};
static alignas(64) uint32_t D[M][N] = {[0 ... M-1][0 ... N-1] = 1};

void rand_init_A() {
    srand((unsigned)time(NULL));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = (uint8_t)(rand() % 256); // 0 ~ 255
        }
    }
}

void rand_init_B() {
    srand((unsigned)time(NULL));
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i][j] = (uint8_t)(rand() % 256); // 0 ~ 255
        }
    }
}

void reorder_A() {
    // int DIM_I = 8, DIM_K = 32;
    int x = 0, y = 0;
    int cnt = 0;
    while(x < M && y < K) {
        for(int i = 0; i < DIM_I; i++) {
            for(int j = 0; j < DIM_K; j++) {
                AR[cnt / K][cnt % K] = A[x + i][y + j];
                cnt++;
            }
        }
        if(y + DIM_K >= K) {
            x += DIM_I, y = 0; 
        }
        else {
            y += DIM_K;
        }
    }
}

void regular_init_A() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = (uint8_t)(i + j);
        }
    }
}

void regular_init_B() {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i][j] = (uint8_t)(i + j);
        }
    }
}

// GEMM golden model: C = A * B + D
static inline void gemm_golden_int32(
    const uint8_t* A, const uint8_t* B, const uint32_t* D, uint32_t* C, 
    size_t D_M, size_t D_N, size_t D_K)
{
    for (size_t i = 0; i < D_M; i++) {
        for (size_t j = 0; j < D_N; j++) {
            uint32_t sum = D ? D[i*D_N + j] : 0;
            for (size_t k = 0; k < D_K; k++) {
                // printf("A[%zu,%zu](A[%zu])=%d, B[%zu,%zu](B[%zu])=%d\n", i, k, i*D_K + k, A[i*D_K + k], k, j, k*D_N + j, B[k*D_N + j]);
                sum += (uint32_t)A[i*D_K + k] * (uint32_t)B[k*D_N + j];
            }
            C[i*D_N + j] = sum;
        }
    }
}

// 比较两个int32矩阵是否完全相等
static inline bool matcmp_int32(const uint32_t* ref, const uint32_t* test, size_t D_M, size_t D_N)
{
    for (size_t i = 0; i < D_M * D_N; i++) {
        if (ref[i] != test[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("start test!\n");
    rand_init_A();
    rand_init_B();
    // regular_init_A();
    // regular_init_B();
    reorder_A();
    printf("Matrix A: %d x %d\n", M, K);
    printf("Matrix B: %d x %d\n", K, N);
    printf("Matrix C: %d x %d\n", M, N);
    printf("Matrix D: %d x %d\n", M, N);
    printf("A addr: 0x%lx-0x%lx\n", (uint64_t)A, (uint64_t)A + sizeof(A) - 1);
    printf("B addr: 0x%lx-0x%lx\n", (uint64_t)B, (uint64_t)B + sizeof(B) - 1);
    printf("C addr: 0x%lx-0x%lx\n", (uint64_t)C, (uint64_t)C + sizeof(C) - 1);
    printf("D addr: 0x%lx-0x%lx\n", (uint64_t)D, (uint64_t)D + sizeof(D) - 1);
    // printf("\nMatrix A:\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < K; j++) {
    //         printf("%4d ", A[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\nMatrix AR:\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < K; j++) {
    //         printf("%4d ", AR[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\nMatrix B:\n");
    // for (int i = 0; i < K; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%4d ", B[i][j]);
    //     }
    //     printf("\n");
    // }
    tiled_matmul_auto(M, N, K,
    (const elem_t*)AR, (const elem_t*)B, (const acc_t*)D, (acc_t*)C,
    K, N, N, N); // stride in elements
    printf("end test!\n");
    // 验证结果
    // printf("\nMatrix A:\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < K; j++) {
    //         printf("%4d ", A[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\nMatrix B:\n");
    // for (int i = 0; i < K; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%4d ", B[i][j]);
    //     }
    //     printf("\n");
    // }
    gemm_golden_int32(A, B, D, C_G, M, N, K);

    printf("\nC:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6d ", C[i][j]);
        }
        printf("\n");
    }
    printf("\nC_GOLDEN:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6d ", C_G[i][j]);
        }
        printf("\n");
    }

    if (matcmp_int32(C_G, C, M, N)) {
        printf("GEMM PASS!!\n");
    } else {
        printf("GEMM FAIL!!\n");
    }

    return 0;
}
