#include "../common/norm_v2.h"
#include <stdio.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>


#define SPM_A_ROWS 2048
#define SPM_B_ROWS 2048
#define SPM_C_ROWS 2048
#define SPM_WIDTH 32 // in bytes
#define BANK_A_START_ADDR 0x0
#define BANK_B_START_ADDR (SPM_A_ROWS * SPM_WIDTH)
#define BANK_C_START_ADDR (SPM_A_ROWS * SPM_WIDTH + SPM_B_ROWS * SPM_WIDTH)

// #define M 16
// #define N 512
// #define M 256
// #define N 768
#define M 256
#define N 64

static alignas(64) int32_t C_S[M][N] = {[0 ... M-1][0 ... N-1] = 1000};
static alignas(64) int32_t C_G[M][N] = {[0 ... M-1][0 ... N-1] = 1};
static alignas(64) int32_t C_T[M][N] = {[0 ... M-1][0 ... N-1] = 1};


static inline void layernorm_golden_int32(
    const int32_t *input,
    int32_t *output,
    size_t D_M,
    size_t D_N,
    double in_scale_factor,
    const size_t intermediate_q_bits)
{
    // === 对称量化 LayerNorm golden model ===
    for (size_t i = 0; i < D_M; i++) {
        const int32_t *row = input + i * D_N;

        // Step 1: 反量化为浮点
        double x_fp[D_N];
        for (size_t j = 0; j < D_N; j++) {
            x_fp[j] = (double)row[j] * in_scale_factor;  // 保留正负号
        }

        // Step 2: 计算均值
        double mean = 0.0;
        for (size_t j = 0; j < D_N; j++)
            mean += x_fp[j];
        mean /= D_N;

        // Step 3: 计算方差
        double var = 0.0;
        for (size_t j = 0; j < D_N; j++) {
            double diff = x_fp[j] - mean;
            var += diff * diff;
        }
        var /= D_N;
        double stddev = sqrt(var + 1e-5);

        // Step 4: 归一化并重新量化
        double out_scale = (double)(1 << 24);  // 输出假设为 Q8.24 可调
        for (size_t j = 0; j < D_N; j++) {
            double norm = (x_fp[j] - mean) / stddev;
            int32_t q = (int32_t)round(norm * out_scale);

            if (q > INT32_MAX)
                q = INT32_MAX;
            else if (q < INT32_MIN)
                q = INT32_MIN;

            output[i * D_N + j] = q;
        }
    }
}


// // 初始化输入矩阵
// void regular_init_C() {
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             C_S[i][j] = (int32_t)((127 * i + i * (( ( i + j ) / 2 * j) % 7) + 10) % 128);
//         }
//     }
// }


void load_C_from_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening input file");
        exit(1);
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fscanf(fp, "%d", &C_S[i][j]) != 1) {
                fprintf(stderr, "Error: not enough data for row %d col %d\n", i, j);
                fclose(fp);
                exit(1);
            }
        }
    }

    fclose(fp);
    printf("Successfully loaded C_S from %s\n", filename);
}



// 比较两个int32矩阵是否完全相等（允许一定容差）
static inline bool matcmp_int32(const int32_t* ref, const int32_t* test, size_t D_M, size_t D_N, size_t tol)
{
    for (size_t i = 0; i < D_M * D_N; i++) {
        if (abs(ref[i] - test[i]) > tol) {
            printf("Mismatch at %zu: ref=%d test=%d\n", i, ref[i], test[i]);
            return false;
        }
    }
    return true;
}

// 行内余弦相似度
double row_cosine_similarity(const int32_t *a, const int32_t *b, size_t D_N) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t j = 0; j < D_N; j++) {
        double va = (double)a[j];
        double vb = (double)b[j];
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0;
    return dot / (sqrt(norm_a) * sqrt(norm_b));
}

// 行内MSE
double row_mse(const int32_t *a, const int32_t *b, size_t D_N) {
    double sum_sq = 0.0;
    for (size_t j = 0; j < D_N; j++) {
        double diff = (double)a[j] - (double)b[j];
        sum_sq += diff * diff;
    }
    return sum_sq / D_N;
}

// 针对LayerNorm的逐行评估
void evaluate_layernorm_per_row(const int32_t *C_T, const int32_t *C_G, size_t D_M, size_t D_N) {
    double total_cos = 0.0;
    double total_mse = 0.0;

    for (size_t i = 0; i < D_M; i++) {
        const int32_t *row_T = C_T + i * D_N;
        const int32_t *row_G = C_G + i * D_N;
        double cos_sim = row_cosine_similarity(row_T, row_G, D_N);
        double mse = row_mse(row_T, row_G, D_N);
        total_cos += cos_sim;
        total_mse += mse;
    }

    double mean_cos = total_cos / D_M;
    double mean_mse = total_mse / D_M;

    printf("=== LayerNorm Row-wise Evaluation ===\n");
    printf("Average Cosine Similarity: %.6f\n", mean_cos);
    printf("Average MSE: %.6e\n", mean_mse);
    if (mean_cos > 0.999)
        printf("Excellent match (shape preserved)\n");
    else if (mean_cos > 0.995)
        printf("Acceptable match\n");
    else
        printf("Potential mismatch or scaling issue\n");
    printf("=====================================\n");
}

int main() {
    printf("=== LayerNorm Test Start ===\n");
    // regular_init_C();
    // load_C_from_file("/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/layernorm/number_quantized.txt");

    // printf("Matrix C: %d x %d\n", M, N);
    // printf("C addr: 0x%lx-0x%lx\n", (uint64_t)C_S, (uint64_t)C_S + sizeof(C_S) - 1);

    // 调用硬件/软件实现版本
    tiled_norm_auto(M, N, (const acc_t*)C_S, (acc_t*)C_T,
                    1.0 / (1 << 0), 0, N, N, LAYERNORM);

    // 计算黄金参考
    // layernorm_golden_int32((const int32_t*)C_S, (int32_t*)C_G, M, N, 1, 0);

    // // 比较结果
    // if (matcmp_int32(C_G, C_T, M, N, 200)) {
    //     printf("LAYERNORM PASS!!\n");
    // } else {
    //     printf("LAYERNORM FAIL!!\n");
    // }

    // 输出输入矩阵
    // printf("=== Input Matrix C_S ===\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%d ", C_S[i][j]);
    //     }
    //     printf("\n");
    // }
    // // 输出前几行对比
    // printf("=== LAYERNORM Test Result (C_T vs C_G) ===\n");
    // for (int i = 0; i < M; i++) {
    //     printf("Row %d\n", i);
    //     for (int j = 0; j < N; j++) {
    //         printf("%d/%d ", C_T[i][j], C_G[i][j]);
    //     }
    //     printf("\n");
    // }
    // evaluate_layernorm_per_row((const int32_t*)C_T, (const int32_t*)C_G, M, N);

    printf("=== Test End ===\n");
    // printf("total operations = %lld\n", norm_v3_cnt);
    return 0;
}

