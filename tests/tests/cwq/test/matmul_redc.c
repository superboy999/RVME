#include "../common/inst.h"
#include "../common/matmul_v2.h"
#include <stdio.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

const int J = 8;

#define M 8
#define N 8
#define K 32
static alignas(64) int32_t A[M][N] = {[0 ... M-1][0 ... N-1] = 1};
static alignas(64) int32_t C[M][N] = {[0 ... M-1][0 ... N-1] = 1};
static alignas(64) int32_t C_G[M][N] = {[0 ... M-1][0 ... N-1] = 0};

void rand_init_A() {
    srand((unsigned)time(NULL));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (int32_t)(0); // -128 ~ 127
            // A[i][j] = (int32_t)(rand() % (1 << 19) - (1 << 20)); // -128 ~ 127
        }
    }
}


void regular_init_A() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (uint8_t)(i + j);
        }
    }
}

// GEMM golden model: C = A * B + D
static inline void mredcadd_golden_int32(
    const int32_t* A, int32_t* C, uint32_t row)
{
    for (size_t j = 0; j < N; j++) {
        int32_t sum = 0;
        for (size_t i = 0; i < M; i++) {
            sum += A[i*N + j];
        }
        C[row*N + j] = sum;
    }
}

static inline void mredcmax_golden_int32(
    const int32_t* A, int32_t* C, uint32_t row)
{
    for (size_t j = 0; j < N; j++) {
        int32_t sum = 1 - (1 << 31);
        for (size_t i = 0; i < M; i++) {
            if(A[i*N + j] > sum) {
                sum = A[i*N + j];
            }
        }
        C[row*N + j] = sum;
    }
}

// GEMM golden model: C = A * B + D
static inline void gemm_golden_int32(
    const int8_t* A, const int8_t* B, const int32_t* D, int32_t* C, 
    size_t D_M, size_t D_N, size_t D_K)
{
    for (size_t i = 0; i < D_M; i++) {
        for (size_t j = 0; j < D_N; j++) {
            int32_t sum = D ? D[i*D_N + j] : 0;
            for (size_t k = 0; k < D_K; k++) {
                // printf("A[%zu,%zu](A[%zu])=%d, B[%zu,%zu](B[%zu])=%d\n", i, k, i*D_K + k, A[i*D_K + k], k, j, k*D_N + j, B[k*D_N + j]);
                sum += (int32_t)A[i*D_K + k] * (int32_t)B[k*D_N + j];
            }
            C[i*D_N + j] = sum;
        }
    }
}


// 比较两个int32矩阵是否完全相等
static inline bool matcmp_int32(const int32_t* ref, const int32_t* test, size_t D_M, size_t D_N, int32_t row)
{
    for (size_t i = row * D_N; i < (row + 1) * D_N; i++) {
        if (ref[i] != test[i]) {
            return false;
        }
    }
    return true;
}

void print_mat() {
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
}

int main() {
    printf("start test!\n");
    rand_init_A();
    // regular_init_A();
    // regular_init_B();

    printf("Matrix A: %d x %d\n", M, K);
    printf("Matrix B: %d x %d\n", K, N);
    printf("Matrix C: %d x %d\n", M, N);
    printf("Matrix D: %d x %d\n", M, N);
    printf("A addr: 0x%lx-0x%lx\n", (uint64_t)A, (uint64_t)A + sizeof(A) - 1);
    printf("C addr: 0x%lx-0x%lx\n", (uint64_t)C, (uint64_t)C + sizeof(C) - 1);
    printf("\nMatrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6d ", A[i][j]);
        }
        printf("\n");
    }
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
    // tiled_matmul_auto(M, N, K,
    // (const elem_t*)AR, (const elem_t*)B, (const acc_t*)D, (acc_t*)C,
    // K, N, N, N); // stride in elements

    msettilemi(M);
    msettileni(N);
    msettileki(K);
    
    dmaload_spm((uint64_t)A, 1024*64, 4);
    msync_spm();
    mlce32_spm(4, 1024*64, 32, 0);
        
    for(int i = 0; i < 8; i++)  {
        asm volatile ("" ::: "memory");
        asm volatile ("fence rw, rw" ::: "memory");
        
        switch (i)
        {
        case 0:
            mredcadd_w_i(6, 4, 0);
            break;
        case 1:
            mredcadd_w_i(6, 4, 1);
            break;
        case 2:
            mredcadd_w_i(6, 4, 2);
            break;
        case 3:
            mredcadd_w_i(6, 4, 3);
            break;
        case 4:
            mredcadd_w_i(6, 4, 4);
            break;
        case 5:
            mredcadd_w_i(6, 4, 5);
            break;
        case 6:
            mredcadd_w_i(6, 4, 6);
            break;
        case 7:
            mredcadd_w_i(6, 4, 7);
            break;        
        default:
            break;
        }
        
        msce32_spm(6, (1024*2)*32, 32, 0);
        msync_spm();
        dmastore_spm((uint64_t)C, (1024*2)*32, 4);
        msync_spm();
        // asm volatile ("" ::: "memory");
        // asm volatile ("fence rw, rw" ::: "memory");
        // msce32_spm(6, (1024*2)*32, 32, 0);
        // msync_spm();
        // dmastore_spm((uint64_t)C, (1024*2)*32, 4);
        // msync_spm();
        // msce32(6, (uint8_t *) C, 32);
        asm volatile ("" ::: "memory");
        asm volatile ("fence rw, rw" ::: "memory");
        mredcadd_golden_int32(A, C_G, i);
        if (matcmp_int32(C_G, C, M, N, i)) {
            printf("GEMM PASS!!\n");
        } else {
            print_mat();
            return 0;
        }
    }

    for(int i = 0; i < 8; i++)  {

        asm volatile ("" ::: "memory");
        asm volatile ("fence rw, rw" ::: "memory");

        // dmaload_spm((uint64_t)A, 0, 1);
        // mlce32_spm(4, 0, 32, 0);
        switch (i)
        {
        case 0:
            mredcmax_w_i(6, 4, 0);
            break;
        case 1:
            mredcmax_w_i(6, 4, 1);
            break;
        case 2:
            mredcmax_w_i(6, 4, 2);
            break;
        case 3:
            mredcmax_w_i(6, 4, 3);
            break;
        case 4:
            mredcmax_w_i(6, 4, 4);
            break;
        case 5:
            mredcmax_w_i(6, 4, 5);
            break;
        case 6:
            mredcmax_w_i(6, 4, 6);
            break;
        case 7:
            mredcmax_w_i(6, 4, 7);
            break;        
        default:
            break;
        }
        msce32_spm(6, (1024*2)*32, 32, 0);
        msync_spm();
        dmastore_spm((uint64_t)C, (1024*2)*32, 1);
        msync_spm(); 
        asm volatile ("" ::: "memory");
        asm volatile ("fence rw, rw" ::: "memory");
        mredcmax_golden_int32(A, C_G, i);
        if (matcmp_int32(C_G, C, M, N, i)) {
            printf("GEMM PASS!!\n");
        } else {
            print_mat();
            return 0;
        }
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
    // gemm_golden_int32(A, B, D, C_G, M, N, K);

    // printf("\nC:\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%6d ", C[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\nC_GOLDEN:\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%6d ", C_G[i][j]);
    //     }
    //     printf("\n");
    // }

    // if (matcmp_int32(C_G, C, M, N)) {
    //     printf("GEMM PASS!!\n");
    // } else {
    //     printf("GEMM FAIL!!\n");
    // }

    return 0;
}
