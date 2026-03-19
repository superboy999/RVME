#include "../common/inst.h"
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
#define M 8
#define N 8
#define K 32
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

void rand_init_C() {
    srand((unsigned)time(NULL));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = (uint32_t)(rand()); // 0 ~ 255
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
    rand_init_C();
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            C_G[i][j] = C[i][j];
        }
    }

    printf("\nC:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6d ", C[i][j]);
        }
        printf("\n");
    }

    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");

    msettilemi(M);
    msettileni(N);

    dmaload_spm((uint64_t)C, 1024*64, 4);
    msync_spm();
    mlce32_spm(6, 1024*64, 32, 0);
    mscte32_spm(6, 1024*64, 32, 0);
    msync_spm();
    dmastore_spm((uint64_t)C, 1024*64, 4);

    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");

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

    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
    
    mlcte32_spm(6, 1024*64, 32, 0);
    msce32_spm(6, 1024*64, 32, 0);
    msync_spm();
    dmastore_spm((uint64_t)C, 1024*64, 4);
    msync_spm();
    asm volatile ("" ::: "memory");
    asm volatile ("fence rw, rw" ::: "memory");
  
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
