#include "../common/norm_v3_2.h"
#include <stdio.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#define CACHELINE_SIZE 64
#define M 256
#define N 256
static alignas(CACHELINE_SIZE) int32_t C_S[M][N] = {[0 ... M-1][0 ... N-1] = 1};
static alignas(CACHELINE_SIZE) int32_t C_G[M][N] = {[0 ... M-1][0 ... N-1] = 1};
static alignas(CACHELINE_SIZE) int32_t C_T[M][N] = {[0 ... M-1][0 ... N-1] = 1};

// GEMM golden model: C_G = softmax(C)
static inline void softmax_golden_int32(
    const int32_t* C, int32_t* C_G, size_t D_M, size_t D_N, size_t input_q_bits, size_t output_q_bits)
{
    double input_scale_factor = (double)(1 << input_q_bits);
    double output_scale_factor = (double)(1 << output_q_bits);
    printf("softmax_golden_int32: D_M=%lu, D_N=%lu, input_scale_factor=%f, output_scale_factor=%f\n", D_M, D_N, input_scale_factor, output_scale_factor);
    for (size_t i = 0; i < D_M; i++) {
        double sum_exp = 0.0;
        double vals[D_N];
        for (size_t j = 0; j < D_N; j++) {
            vals[j] = (double)C[i * D_N + j] / input_scale_factor;
        }
        // printf("Row %lu: \n", i);
        // for (size_t j = 0; j < D_N; j++) {
        //     printf("%f ", vals[j]);
        // }
        // printf("\n");
        // find max
        double max_val = vals[0];
        for (size_t j = 0; j < D_N; j++) {
            if (vals[j] > max_val) {
                max_val = vals[j];
            }
        }
        // compute exp and sum
        for (size_t j = 0; j < D_N; j++) {
            vals[j] = exp(vals[j] - max_val);
            sum_exp += vals[j];
        }
        // for (size_t j = 0; j < D_N; j++) {
        //     printf("%f ", vals[j]);
        // }
        // printf("\n");
        // normalize
        for (size_t j = 0; j < D_N; j++) {
            vals[j] = vals[j] / sum_exp;
            C_G[i * D_N + j] = (int32_t)(vals[j] * output_scale_factor);
        }
        // for (size_t j = 0; j < D_N; j++) {
        //     printf("%f ", vals[j]);
        // }
        // printf("\n");
        // for (size_t j = 0; j < D_N; j++) {
        //     printf("%d ", C_G[i * D_N + j]);
        // }
        // printf("\n");
    }
}

void regular_init_C() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_S[i][j] = (int32_t)(10*i + j*100);
        }
    }
}

// 比较两个int32矩阵是否相等
static inline bool matcmp_int32(const int32_t* ref, const int32_t* test, size_t D_M, size_t D_N, size_t tol)
{
    for (size_t i = 0; i < D_M * D_N; i++) {
        if (abs(ref[i] - test[i]) > tol) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("start test!\n");
    // regular_init_C();
    // printf("Matrix C: %d x %d\n", M, N);
    // printf("C addr: 0x%lx-0x%lx\n", (uint64_t)C_S, (uint64_t)C_S + sizeof(C_S) - 1);
    tiled_norm_auto(M, N, (const acc_t*)C_S, (acc_t*)C_T, 1.0 / (1<<8), 16, N, N, SOFTMAX); // stride in elements
    printf("end test!\n");
    // 验证结果
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%d ", C_S[i][j]);
    //     }
    //     printf("\n");
    // }
    // softmax_golden_int32((const int32_t*)C_S, (int32_t*)C_G, M, N, 8, 16);
    // if (matcmp_int32((const int32_t*)C_G, (const int32_t*)C_T, M, N, 2000)) {
    //     printf("SOFTMAX PASS!!\n");
    // } else {
    //     printf("SOFTMAX FAIL!!\n");
    // }
    // printf("GEMM test result C_T:\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%d ", C_T[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("GEMM golden:\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%d ", C_G[i][j]);
    //     }
    //     printf("\n");
    // }

    // printf("total operations = %lld\n", norm_v3_cnt);
    
    return 0;
}
