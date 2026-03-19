// #include "../include/inst.h"
// #include "../include/softmax.h"
// #include <stdio.h>
// #include <stdalign.h>
// #include <stddef.h>
// #include <stdbool.h>
// #include <math.h>

// #define SPM_A_ROWS 1024
// #define SPM_B_ROWS 1024
// #define SPM_C_ROWS 2048
// #define SPM_WIDTH 32 // in bytes
// #define BANK_A_START_ADDR 0x0
// #define BANK_B_START_ADDR (SPM_A_ROWS * SPM_WIDTH)
// #define BANK_C_START_ADDR (SPM_A_ROWS * SPM_WIDTH + SPM_B_ROWS * SPM_WIDTH)

// #define M 768
// #define N 768
// // #define K 768
// // #define M 32
// // #define N 32
// // #define K 32
// static alignas(32) int32_t C_S[M][N] = {[0 ... M-1][0 ... N-1] = 0};
// static alignas(32) int32_t C_G[M][N] = {[0 ... M-1][0 ... N-1] = 0};
// static alignas(32) int32_t C_T[M][N] = {[0 ... M-1][0 ... N-1] = 0};

// // GEMM golden model: C_G = softmax(C)
// static inline void softmax_golden_int32(
//     const int32_t* C, int32_t* C_G, size_t D_M, size_t D_N, size_t input_q_bits, size_t output_q_bits)
// {
//     double input_scale_factor = (double)(1 << input_q_bits);
//     double output_scale_factor = (double)(1 << output_q_bits);
//     printf("softmax_golden_int32: D_M=%lu, D_N=%lu, input_scale_factor=%f, output_scale_factor=%f\n", D_M, D_N, input_scale_factor, output_scale_factor);
//     for (size_t i = 0; i < D_M; i++) {
//         double sum_exp = 0.0;
//         double vals[D_N];
//         for (size_t j = 0; j < D_N; j++) {
//             vals[j] = (double)C[i * D_N + j] / input_scale_factor;
//         }
//         printf("Row %lu: \n", i);
//         for (size_t j = 0; j < D_N; j++) {
//             printf("%f ", vals[j]);
//         }
//         printf("\n");
//         // find max
//         double max_val = vals[0];
//         for (size_t j = 0; j < D_N; j++) {
//             if (vals[j] > max_val) {
//                 max_val = vals[j];
//             }
//         }
//         // compute exp and sum
//         for (size_t j = 0; j < D_N; j++) {
//             vals[j] = exp(vals[j] - max_val);
//             sum_exp += vals[j];
//         }
//         for (size_t j = 0; j < D_N; j++) {
//             printf("%f ", vals[j]);
//         }
//         printf("\n");
//         // normalize
//         for (size_t j = 0; j < D_N; j++) {
//             vals[j] = vals[j] / sum_exp;
//             C_G[i * D_N + j] = (int32_t)(vals[j] * output_scale_factor);
//         }
//         for (size_t j = 0; j < D_N; j++) {
//             printf("%f ", vals[j]);
//         }
//         printf("\n");
//         for (size_t j = 0; j < D_N; j++) {
//             printf("%d ", C_G[i * D_N + j]);
//         }
//         printf("\n");
//     }
// }



// void regular_init_C() {
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             C_S[i][j] = (int32_t)(10*i + j * 1000);
//         }
//     }
// }

// // 比较两个int32矩阵是否完全相等
// static inline bool matcmp_int32(const int32_t* ref, const int32_t* test, size_t D_M, size_t D_N, size_t tol)
// {
//     for (size_t i = 0; i < D_M * D_N; i++) {
//         if (abs(ref[i] - test[i]) > tol) {
//             return false;
//         }
//     }
//     return true;
// }

// int main() {
//     printf("start test!\n");
//     regular_init_C();
//     printf("Matrix C: %d x %d\n", M, N);
//     printf("C addr: 0x%lx-0x%lx\n", (uint64_t)C_S, (uint64_t)C_S + sizeof(C_S) - 1);
//     tiled_norm_auto(M, N, (const acc_t*)C_S, (acc_t*)C_T, 1.0 / (1<<8), 16, N, N, SOFTMAX); // stride in elements
//     printf("end test!\n");
//     // 验证结果
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             printf("%d ", C_S[i][j]);
//         }
//         printf("\n");
//     }
//     softmax_golden_int32((const int32_t*)C_S, (int32_t*)C_G, M, N, 8, 16);
//     if (matcmp_int32(C_G, C_T, M, N, 200)) {
//         printf("GEMM PASS!!\n");
//     } else {
//         printf("GEMM FAIL!!\n");
//     }
//     // msettilem(8);
//     // msettilek(32);
//     // msettilen(8);
//     // mlae8(0, (uint64_t)A, 32);
//     // mlae8(1, (uint64_t)B, 32);
//     // mmacc_w_b(4, 0, 1);
//     // dmaload_spm((uint64_t)C, BANK_C_START_ADDR, 8); // load C to SPM_C
//     // mlce32_spm(5, BANK_C_START_ADDR, 32, 0); // load C from SPM_C to acc_reg[5]
//     // madd_w_mm(5, 4, 5);
//     // // msae8_spm(0, BANK_A_START_ADDR, 32, 0); // mode = 0 for SPM_A to SPM_B
//     // // mlae8_spm(0, BANK_A_START_ADDR, 32, 0); // mode = 0 for SPM_A to SPM_B
//     // // msce32_spm(5, BANK_C_START_ADDR, 32, 0); // mode = 0 for SPM_A to SPM_B
//     // msce32_spm(5, BANK_C_START_ADDR, 32, 0); // mode = 0 for SPM_A to SPM_B
//     // dmastore_spm((uint64_t)C, BANK_C_START_ADDR, 8); // store acc_reg[5] to C
//     printf("GEMM test result C_T:\n");
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             printf("%d ", C_T[i][j]);
//         }
//         printf("\n");
//     }
//     printf("GEMM golden:\n");
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             printf("%d ", C_G[i][j]);
//         }
//         printf("\n");
//     }

//     return 0;
// }

// -------------------------------------------- layernorm test ------------------------------------------

// #include "../include/inst.h"
// #include "../include/softmax.h"
// #include <stdio.h>
// #include <stdalign.h>
// #include <stddef.h>
// #include <stdbool.h>
// #include <math.h>
// #include <stdlib.h>


// #define SPM_A_ROWS 1024
// #define SPM_B_ROWS 1024
// #define SPM_C_ROWS 2048
// #define SPM_WIDTH 32 // in bytes
// #define BANK_A_START_ADDR 0x0
// #define BANK_B_START_ADDR (SPM_A_ROWS * SPM_WIDTH)
// #define BANK_C_START_ADDR (SPM_A_ROWS * SPM_WIDTH + SPM_B_ROWS * SPM_WIDTH)

// #define M 16
// #define N 512
// // #define M 32
// // #define N 32

// static alignas(32) int32_t C_S[M][N] = {[0 ... M-1][0 ... N-1] = 0};
// static alignas(32) int32_t C_G[M][N] = {[0 ... M-1][0 ... N-1] = 0};
// static alignas(32) int32_t C_T[M][N] = {[0 ... M-1][0 ... N-1] = 0};


// static inline void layernorm_golden_int32(
//     const int32_t *input,
//     int32_t *output,
//     size_t D_M,
//     size_t D_N,
//     double in_scale_factor,
//     const size_t intermediate_q_bits)
// {
//     // === 对称量化 LayerNorm golden model ===
//     for (size_t i = 0; i < D_M; i++) {
//         const int32_t *row = input + i * D_N;

//         // Step 1: 反量化为浮点
//         double x_fp[D_N];
//         for (size_t j = 0; j < D_N; j++) {
//             x_fp[j] = (double)row[j] * in_scale_factor;  // 保留正负号
//         }

//         // Step 2: 计算均值
//         double mean = 0.0;
//         for (size_t j = 0; j < D_N; j++)
//             mean += x_fp[j];
//         mean /= D_N;

//         // Step 3: 计算方差
//         double var = 0.0;
//         for (size_t j = 0; j < D_N; j++) {
//             double diff = x_fp[j] - mean;
//             var += diff * diff;
//         }
//         var /= D_N;
//         double stddev = sqrt(var + 1e-5);

//         // Step 4: 归一化并重新量化
//         double out_scale = (double)(1 << 24);  // 输出假设为 Q8.24 可调
//         for (size_t j = 0; j < D_N; j++) {
//             double norm = (x_fp[j] - mean) / stddev;
//             int32_t q = (int32_t)round(norm * out_scale);

//             if (q > INT32_MAX)
//                 q = INT32_MAX;
//             else if (q < INT32_MIN)
//                 q = INT32_MIN;

//             output[i * D_N + j] = q;
//         }
//     }
// }


// // // 初始化输入矩阵
// // void regular_init_C() {
// //     for (int i = 0; i < M; i++) {
// //         for (int j = 0; j < N; j++) {
// //             C_S[i][j] = (int32_t)((127 * i + i * (( ( i + j ) / 2 * j) % 7) + 10) % 128);
// //         }
// //     }
// // }


// void load_C_from_file(const char *filename) {
//     FILE *fp = fopen(filename, "r");
//     if (!fp) {
//         perror("Error opening input file");
//         exit(1);
//     }

//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             if (fscanf(fp, "%d", &C_S[i][j]) != 1) {
//                 fprintf(stderr, "Error: not enough data for row %d col %d\n", i, j);
//                 fclose(fp);
//                 exit(1);
//             }
//         }
//     }

//     fclose(fp);
//     printf("✅ Successfully loaded C_S from %s\n", filename);
// }



// // 比较两个int32矩阵是否完全相等（允许一定容差）
// static inline bool matcmp_int32(const int32_t* ref, const int32_t* test, size_t D_M, size_t D_N, size_t tol)
// {
//     for (size_t i = 0; i < D_M * D_N; i++) {
//         if (abs(ref[i] - test[i]) > tol) {
//             printf("Mismatch at %zu: ref=%d test=%d\n", i, ref[i], test[i]);
//             return false;
//         }
//     }
//     return true;
// }

// // 行内余弦相似度
// double row_cosine_similarity(const int32_t *a, const int32_t *b, size_t D_N) {
//     double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
//     for (size_t j = 0; j < D_N; j++) {
//         double va = (double)a[j];
//         double vb = (double)b[j];
//         dot += va * vb;
//         norm_a += va * va;
//         norm_b += vb * vb;
//     }
//     if (norm_a == 0.0 || norm_b == 0.0) return 0.0;
//     return dot / (sqrt(norm_a) * sqrt(norm_b));
// }

// // 行内MSE
// double row_mse(const int32_t *a, const int32_t *b, size_t D_N) {
//     double sum_sq = 0.0;
//     for (size_t j = 0; j < D_N; j++) {
//         double diff = (double)a[j] - (double)b[j];
//         sum_sq += diff * diff;
//     }
//     return sum_sq / D_N;
// }

// // 针对LayerNorm的逐行评估
// void evaluate_layernorm_per_row(const int32_t *C_T, const int32_t *C_G, size_t D_M, size_t D_N) {
//     double total_cos = 0.0;
//     double total_mse = 0.0;

//     for (size_t i = 0; i < D_M; i++) {
//         const int32_t *row_T = C_T + i * D_N;
//         const int32_t *row_G = C_G + i * D_N;
//         double cos_sim = row_cosine_similarity(row_T, row_G, D_N);
//         double mse = row_mse(row_T, row_G, D_N);
//         total_cos += cos_sim;
//         total_mse += mse;
//     }

//     double mean_cos = total_cos / D_M;
//     double mean_mse = total_mse / D_M;

//     printf("=== LayerNorm Row-wise Evaluation ===\n");
//     printf("Average Cosine Similarity: %.6f\n", mean_cos);
//     printf("Average MSE: %.6e\n", mean_mse);
//     if (mean_cos > 0.999)
//         printf("✅ Excellent match (shape preserved)\n");
//     else if (mean_cos > 0.995)
//         printf("✅ Acceptable match\n");
//     else
//         printf("⚠️  Potential mismatch or scaling issue\n");
//     printf("=====================================\n");
// }

// int main() {
//     printf("=== LayerNorm Test Start ===\n");
//     // regular_init_C();
//     load_C_from_file("/cluster/home/geyh/RV/xuantie_gnu_toolchain/test/rvme/tests/number_quantized.txt");

//     printf("Matrix C: %d x %d\n", M, N);
//     printf("C addr: 0x%lx-0x%lx\n", (uint64_t)C_S, (uint64_t)C_S + sizeof(C_S) - 1);

//     // 调用硬件/软件实现版本
//     tiled_norm_auto(M, N, (const acc_t*)C_S, (acc_t*)C_T,
//                     1.0 / (1 << 0), 0, N, N, LAYERNORM);

//     // 计算黄金参考
//     layernorm_golden_int32((const int32_t*)C_S, (int32_t*)C_G, M, N, 1, 0);

//     // // 比较结果
//     // if (matcmp_int32(C_G, C_T, M, N, 200)) {
//     //     printf("LAYERNORM PASS!!\n");
//     // } else {
//     //     printf("LAYERNORM FAIL!!\n");
//     // }

//     // 输出输入矩阵
//     printf("=== Input Matrix C_S ===\n");
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             printf("%d ", C_S[i][j]);
//         }
//         printf("\n");
//     }
//     // 输出对比
//     printf("=== LAYERNORM Test Result (C_T vs C_G) ===\n");
//     for (int i = 0; i < M; i++) {
//         printf("Row %d\n", i);
//         for (int j = 0; j < N; j++) {
//             printf("%d/%d ", C_T[i][j], C_G[i][j]);
//         }
//         printf("\n");
//     }
//     evaluate_layernorm_per_row((const int32_t*)C_T, (const int32_t*)C_G, M, N);

//     printf("=== Test End ===\n");

//     return 0;
// }



//-------------------------------------IGELU test----------------------------------------
#include "../common/inst.h"
#include "../common/norm.h"
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdio.h>


#define SPM_A_ROWS 1024
#define SPM_B_ROWS 1024
#define SPM_C_ROWS 2048
#define SPM_WIDTH 32 // in bytes
#define BANK_A_START_ADDR 0x0
#define BANK_B_START_ADDR (SPM_A_ROWS * SPM_WIDTH)
#define BANK_C_START_ADDR (SPM_A_ROWS * SPM_WIDTH + SPM_B_ROWS * SPM_WIDTH)


#define M 32
#define N 32
// #define M 32
// #define N 32

static alignas(32) int32_t C_S[M][N] = {[0 ... M-1][0 ... N-1] = 0};
static alignas(32) int32_t C_G[M][N] = {[0 ... M-1][0 ... N-1] = 0};
static alignas(32) int32_t C_T[M][N] = {[0 ... M-1][0 ... N-1] = 0};

/* 类型别名：根据你的实现调整 */
typedef int32_t acc_t;        // 输入：Q32.0 (integer)
typedef int32_t q16_t;        // 输出：Q16.16 fixed-point stored in int32_t
typedef float   acc_scale_t;  // bert_scale as float

/* helper: round-to-nearest-even for double -> long long */
static inline long long round_near_even_double_to_ll(double v) {
    double r = floor(v);
    double frac = v - r;

    if (frac > 0.5) return (long long)(r + 1.0);
    if (frac < 0.5) return (long long)r;
    // exactly 0.5 -> round to even
    long long ri = (long long)r;
    return ( (ri & 1LL) == 0 ? ri : ri + 1LL );
}

/*
 * igelu_golden_q32_to_q16
 *  - 输入 x_q32: acc_t 表示 Q32.0（即一个整数）
 *  - bert_scale: 与 Gemmini 中同名含义，用于计算 qb,qc,q1（float）
 *  - 返回 int32_t，表示 Q16.16 固定点值 (真实值 * 2^16, rounded-to-even)
 */
q16_t igelu_golden_q32_to_q16(acc_t x_q32, acc_scale_t scale, acc_scale_t bert_scale) {
    /* 常数（与 Gemmini 一致） */
    const acc_scale_t sqrt_2 = 1.41421356237f;
    const acc_scale_t S = bert_scale;

    /* 计算 q1, qb, qc（保持浮点形式以便拟合常数） */
    const acc_scale_t S_over_sqrt2 = S / sqrt_2;
    const acc_scale_t S_erf = (-0.2888f * S_over_sqrt2 * S_over_sqrt2);

    const acc_scale_t q1_f = 1.0f / S_erf;                          // float
    const acc_scale_t qb_f = -1.769f / S_over_sqrt2;                 // float
    const acc_scale_t qc_f = 1.0f / (-0.2888f * S_over_sqrt2 * S_over_sqrt2);

    /* 把输入 integer interpret 为浮点数（Q32.0 -> float 值相等） */
    double q = (double)x_q32;  // use double for slightly more precision

    /* sign, abs, clip (按 Gemmini 实现) */
    double q_sign = (q < 0.0) ? -1.0 : 1.0;
    double q_abs = q < 0.0 ? -q : q;

    /* clip 到 -qb（注意 qb_f 为负数），Gemmini 使用 abs(q) > (-qb) ? (-qb) : abs(q) */
    double clip_limit = -(double)qb_f; // -qb_f is positive
    double q_clipped = q_abs > clip_limit ? clip_limit : q_abs;

    /* 多项式 (q_clipped + qb)^2 + qc  —— 注意 qb 是负的，和原实现一致 */
    double tmp = (q_clipped + (double)qb_f);
    double q_poly = tmp * tmp + (double)qc_f;

    /* erf 近似，恢复符号 */
    double q_erf = q_sign * q_poly;

    /* 最终 IGELU 双精度结果（和原代码 x = q * (q_erf + q1) 对应） */
    double y = q * (q_erf + (double)q1_f);

    y = y * scale;

    long long y_rounded = round_near_even_double_to_ll(y);

    /* clip 到 int32_t 能表示的范围（因为我们存 Q16.16 于 int32_t） */
    if (y_rounded > INT32_MAX) y_rounded = INT32_MAX;
    if (y_rounded < INT32_MIN) y_rounded = INT32_MIN;

    return (acc_t) y_rounded;
}

q16_t gelu_golden_q32_to_q16(acc_t x_q32, acc_scale_t scale, acc_scale_t bert_scale)
{
    /* 常数 */
    const double sqrt_2 = 1.4142135623730951;

    /* 输入 interpret 为 double（保持与 igelu 一致） */
    double x = (double)x_q32 / (double)bert_scale;  // 先反量化到 float 值

    /* GELU(x) = 0.5 * x * (1 + erf( x / sqrt(2) )) */
    double t = x / sqrt_2;
    double erf_val = erf(t);
    double y = 0.5 * x * (1.0 + erf_val);

    /* 乘 scale（与 igelu 同结构） */
    y = y * (double)scale;

    long long y_rounded = round_near_even_double_to_ll(y);

    /* clip 到 int32 范围 */
    if (y_rounded > INT32_MAX) y_rounded = INT32_MAX;
    if (y_rounded < INT32_MIN) y_rounded = INT32_MIN;

    return (acc_t) y_rounded;
}


static acc_t scale_and_sat(acc_t x, int act, acc_scale_t scale, acc_scale_t bert_scale) {
  // Apply I-GELU if needed
  if (act == IGELU) {
    const acc_scale_t sqrt_2 = 1.41421356237;

    const acc_scale_t S = bert_scale;

    const acc_scale_t S_erf = (-0.2888 * (S/sqrt_2)*(S/sqrt_2));  //-5.776 * 10e-5
    const acc_t q1 = 1 / S_erf; // -17313.01939 
    const acc_t qb = -1.769 / (S / sqrt_2); //-125.0871896
    const acc_t qc = 1.0 / (-0.2888 * (S / sqrt_2) * (S / sqrt_2)); //-17313.01939

    printf("q1: %d, qb: %d, qc: %d\n", q1, qb, qc);

    const acc_t q = x;

    const acc_t q_sign = q < 0 ? -1 : 1;
    const acc_t q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
    const acc_t q_poly = (q_clipped + qb)*(q_clipped + qb) + qc;
    const acc_t q_erf = q_sign * q_poly;
    printf("(q+b)^2: %d\n", (q_clipped + qb)*(q_clipped + qb));
    printf("q_poly: %d, q_erf: %d\n", q_poly, q_erf);  
    x = q * (q_erf + q1);
    printf("IGELU output (Q32.0) x = q * (q_erf + q1): %d = %d *(%d + %d)\n", x, q, q_erf, q1);

    printf("IGELU output (Q16.16): %d\n", x);
  }

//   // Scale value down and round it
//   x = ACC_SCALE(x, scale);
  // Clip result
//   x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
//   x = x * 65536; // scale to Q16.16
//   // Apply activation function
//   if (act == RELU) {
//     x = x < 0 ? 0 : x;
//   }
  return x;
}

void golden_IGELU_matrix(
    int32_t C_in[M][N],      // 输入：Q32.0
    int32_t C_out[M][N],     // 输出：Q16.16
    float scale,
    float bert_scale
){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            // C_out[i][j] = scale_and_sat(C_in[i][j], IGELU, 1.0f, bert_scale);
            // C_out[i][j] = igelu_golden_q32_to_q16(C_in[i][j], scale, bert_scale);
            C_out[i][j] = gelu_golden_q32_to_q16(C_in[i][j], scale, bert_scale);
        }
    }
}


//=====================
// GELU golden model（true GELU）
// x input: float
// return: float
//=====================
float gelu_golden_float(float x) {
    return 0.5f * x * (1.0f + erff(x / 1.41421356237f));
}


//========================================
// 主测试函数
// 输入：你的矩阵(M×N)，你的 IGELU 输出矩阵，测试结果打印
//========================================
void evaluate_gelu_results() {
    double mse = 0.0;
    double mae = 0.0;
    int32_t max_err = 0;
    int64_t count = (int64_t)M * (int64_t)N;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t g = C_G[i][j];
            int32_t t = C_T[i][j];
            int32_t err = t - g;

            mse += (double)err * err;
            mae += fabs((double)err);
            if (abs(err) > max_err) max_err = abs(err);
        }
    }

    mse /= (double)count;
    mae /= (double)count;

    printf("========== GELU Approximation Quality ==========\n");
    printf("MSE       = %.6f\n", mse);
    printf("MAE       = %.6f\n", mae);
    printf("Max Error = %d\n", max_err);
    printf("===============================================\n");
}


// 初始化输入矩阵
void regular_init_C() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_S[i][j] = (int32_t)((127 * i + i * (( ( i + j ) / 2 * j) % 7) + 10) % 128);
            // C_S[i][j] = (int32_t)(10 * i + j);
        }
    }
}

void change_C_T (acc_scale_t bert_scale) {
    double scale = (-0.2888 * bert_scale) / 4; // 1 / (2q1 * bert_scale)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_T[i][j] = (int32_t)(C_T[i][j] * scale);
        }
    }   
}

int main(){
    printf("=== IGELU Test Start ===\n");
    regular_init_C();

    printf("Matrix C: %d x %d\n", M, N);
    printf("C addr: 0x%lx-0x%lx\n", (uint64_t)C_S, (uint64_t)C_S + sizeof(C_S) - 1);

    // 调用硬件/软件实现版本
    tiled_norm_auto(M, N, (const acc_t*)C_S, (acc_t*)C_T,
                    0.02 / (1 << 0), 0, N, N, IGELU);

    // 计算黄金参考
    golden_IGELU_matrix(C_S, C_G, 1.0f ,0.02f);

    change_C_T(0.02f);

    evaluate_gelu_results();

    // 输出输入矩阵
    printf("=== Input Matrix C_S ===\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C_S[i][j]);
        }
        printf("\n");
    }
    // 输出对比
    printf("=== IGELU Test Result (C_T vs C_G) ===\n");
    for (int i = 0; i < M; i++) {
        printf("Row %d\n", i);
        for (int j = 0; j < N; j++) {
            printf("%d/%d ", C_T[i][j], C_G[i][j]);
        }
        printf("\n");
    }

    printf("=== Test End ===\n");


    // printf("scale_and_sat test:\n");
    // acc_t test_val = 100;
    // acc_scale_t test_scale = 1.0f;
    // acc_scale_t test_bert_scale = 0.02f;
    // elem_t result = gelu_golden_q32_to_q16(test_val, test_scale, test_bert_scale);
    // printf("Result: %d\n", result); 


    return 0;
}
