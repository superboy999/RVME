#include "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/common/matmul_v2_2_2.h"
#include "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/common/norm_v3_2.h"
#include "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/common/matadd_v2_2.h"
#include "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/common/fileio.h"
// #include "../include/layernorm_v2.h"
#include <stdio.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define CACHELINE_SIZE 64
#define SEQ_LEN 128
#define HIDDEN_DIM 768
#define NUM_HEADS 12

static alignas(CACHELINE_SIZE) elem_t input[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t Wq[HIDDEN_DIM][HIDDEN_DIM] = {[0 ... HIDDEN_DIM-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t Wk[HIDDEN_DIM][HIDDEN_DIM] = {[0 ... HIDDEN_DIM-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t Wv[HIDDEN_DIM][HIDDEN_DIM] = {[0 ... HIDDEN_DIM-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t Wq_b[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t Wk_b[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t Wv_b[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t Q_buf[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t K_buf[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t V_buf[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t Q_in[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t KT_in[HIDDEN_DIM][SEQ_LEN] = {[0 ... HIDDEN_DIM-1][0 ... SEQ_LEN-1] = 1};
static alignas(CACHELINE_SIZE) acc_t attn_buf[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {[0 ... NUM_HEADS-1][0 ... SEQ_LEN-1][0 ... SEQ_LEN-1] = 1};
static alignas(CACHELINE_SIZE) acc_t softmax_in[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {[0 ... NUM_HEADS-1][0 ... SEQ_LEN-1][0 ... SEQ_LEN-1] = 1};
static alignas(CACHELINE_SIZE) acc_t softmax_attn_buf[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {[0 ... NUM_HEADS-1][0 ... SEQ_LEN-1][0 ... SEQ_LEN-1] = 1};
static alignas(CACHELINE_SIZE) elem_t attn_in[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {[0 ... NUM_HEADS-1][0 ... SEQ_LEN-1][0 ... SEQ_LEN-1] = 1};
static alignas(CACHELINE_SIZE) elem_t V_in[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t out_buf[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t out_in[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t Wo[HIDDEN_DIM][HIDDEN_DIM] = {[0 ... HIDDEN_DIM-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t Wo_b[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t out_buf_acc[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t resadd_input[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t resadd_out[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t LN_in[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t out[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};

// static alignas(CACHELINE_SIZE) acc_t Q_buf_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
// static alignas(CACHELINE_SIZE) acc_t K_buf_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
// static alignas(CACHELINE_SIZE) acc_t V_buf_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
// static alignas(CACHELINE_SIZE) acc_t attn_buf_cp[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {[0 ... NUM_HEADS-1][0 ... SEQ_LEN-1][0 ... SEQ_LEN-1] = 1};
// static alignas(CACHELINE_SIZE) acc_t softmax_attn_buf_cp[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {[0 ... NUM_HEADS-1][0 ... SEQ_LEN-1][0 ... SEQ_LEN-1] = 1};
// static alignas(CACHELINE_SIZE) acc_t out_buf_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
// static alignas(CACHELINE_SIZE) acc_t out_buf_acc_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
// static alignas(CACHELINE_SIZE) acc_t resadd_out_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
// static alignas(CACHELINE_SIZE) acc_t out_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};

static alignas(CACHELINE_SIZE) acc_t Q_buf_g[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t K_buf_g[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t V_buf_g[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t attn_buf_g[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {[0 ... NUM_HEADS-1][0 ... SEQ_LEN-1][0 ... SEQ_LEN-1] = 1};


static alignas(CACHELINE_SIZE) elem_t inputR[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t Q_inR[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};

// layernorm

void load_C_from_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening input file");
        exit(1);
    }

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            if (fscanf(fp, "%d", &LN_in[i][j]) != 1) {
                fprintf(stderr, "Error: not enough data for row %d col %d\n", i, j);
                fclose(fp);
                exit(1);
            }
        }
    }

    fclose(fp);
    printf("Successfully loaded C_S from %s\n", filename);
}



void attention(int hidden_dim, int num_heads, int seq_len,
        const elem_t * input, const elem_t * enc_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv,
        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        acc_t * Q_buf, acc_t * K_buf, acc_t * V_buf,

        const elem_t * Q_in, const elem_t * KT_in,
        acc_t * attn_buf, 
        const acc_t * softmax_in, acc_t * softmax_attn_buf,

        const elem_t * attn_in, const elem_t * V_in,
        acc_t * out_buf,

        const elem_t * out_in, const elem_t * Wo, const acc_t * Wo_b,
        acc_t * out_buf_acc,

        const acc_t * resadd_input,
        acc_t * resadd_out, 
        const acc_t * LN_in,
        acc_t * out
)
{
    int hidden_dim_per_head = hidden_dim / num_heads; // 64

    // Q = Wq * input + Wq_b
    // K = Wk * enc_out + Wk_b
    // V = Wv * enc_out + Wv_b
    const int qkv_matmuls_n = 3;
    for (int i = 0; i < qkv_matmuls_n; i++) {
        const elem_t * qkv_weights[] = {Wq, Wk, Wv};
        const elem_t * qkv_ins[] = {input, enc_out, enc_out};
        const acc_t * qkv_bs[] = {Wq_b, Wk_b, Wv_b};
        acc_t * qkv_outs[] = {Q_buf, K_buf, V_buf};
        const elem_t * qkv_w = qkv_weights[i];
        const elem_t * qkv_in = qkv_ins[i];
        const acc_t * qkv_b = qkv_bs[i];
        acc_t * qkv_out = qkv_outs[i];
        tiled_matmul_auto(seq_len, hidden_dim, hidden_dim, qkv_in, 
            qkv_w, qkv_b, qkv_out, hidden_dim, hidden_dim, hidden_dim, hidden_dim);
    }

    printf("QKV matmuls done.\n");
    
    // // attn = Q * K^T
    // for (int head = 0; head < num_heads; head++) {
    //     const elem_t * A = Q_in + head * hidden_dim_per_head;
    //     const elem_t * B = KT_in + head * hidden_dim_per_head * seq_len;
    //     acc_t * C = attn_buf + head * seq_len * seq_len;
    //     tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head, A, B, NULL, C,
    //         hidden_dim, seq_len, 0, seq_len);
    // }
    // printf("QK^T finished\n");

    // // attn = softmax(attn)
    // for (int head = 0; head < num_heads; head++) {
    //     const acc_t * in = softmax_in + head * seq_len * seq_len;
    //     acc_t * out = softmax_attn_buf + head * seq_len * seq_len;
    //     tiled_norm_auto(seq_len, seq_len, in, out,
    //         1.0 / (1<<8), 16, seq_len, seq_len, SOFTMAX);
    // }
    // printf("softmax finished\n");

    // // out_buf = attn * V
    // for (int head = 0; head < num_heads; head++) {
    //     const elem_t * A = attn_in + head * seq_len * seq_len;
    //     const elem_t * B = V_in + head * hidden_dim_per_head;
    //     acc_t * C = out_buf + head * hidden_dim_per_head;
    //     tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len, A, B, NULL, C,
    //         seq_len, hidden_dim, 0, hidden_dim);
    // }
    // printf("attn * V finished\n");

    // // out_buf_acc = out_buf * Wo + Wo_b
    // tiled_matmul_auto(/*DIM_I=*/seq_len, /*DIM_J=*/hidden_dim, /*DIM_K=*/hidden_dim,
    //     /*A=*/ out_in, /*B=*/ Wo,
    //     /*D=*/ Wo_b, /*C=*/ out_buf_acc,
    //     /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/hidden_dim, /*stride_C=*/hidden_dim);
    // printf("output matmul finished\n");

    // // resadd_out = out_buf_acc + input
    // tiled_add_auto(seq_len, hidden_dim,
    //     out_buf_acc, resadd_input, resadd_out, hidden_dim, hidden_dim, hidden_dim);
    // // printf("resadd_out printed\n");

    // for (int i = 0; i < SEQ_LEN; i++) {
    //     for(int j = 0; j < HIDDEN_DIM; j++) {
    //         printf("%d ", resadd_out[i * HIDDEN_DIM + j]);
    //     }
    //     printf("\n");
    // }

    // // out = LN(resadd_out)
    // tiled_norm_auto(seq_len, hidden_dim, LN_in, out,
    //         1.0 / (1<<0), 0, seq_len, hidden_dim, LAYERNORM);
    // printf("LayerNorm done.\n");
}

/* ---------------------------------------------------  */
// matmul test begin
/* ---------------------------------------------------  */

void rand_init_matmul() {
    srand((unsigned)time(NULL));
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            // input[i][j] = (int8_t)(rand() % 256 - 128); // -128 ~ 127
            input[i][j] = (int8_t)((i + j)% 256 - 128); // -128 ~ 127
        }
    }

    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            // Wq[i][j] = (int8_t)(rand() % 256 - 128); // -128 ~ 127
            Wq[i][j] = (int8_t)((i + j)% 256 - 128); // -128 ~ 127
        }
    }

    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            // Wk[i][j] = (int8_t)(rand() % 256 - 128); // -128 ~ 127
            Wk[i][j] = (int8_t)((i + j)% 256 - 128); // -128 ~ 127
        }
    }

    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            // Wv[i][j] = (int8_t)(rand() % 256 - 128); // -128 ~ 127
            Wv[i][j] = (int8_t)((i + j)% 256 - 128); // -128 ~ 127
        }
    }

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            // Wq_b[i][j] = (int32_t)(rand() % 256 - 128); // -128 ~ 127
            Wq_b[i][j] = (int32_t)((i + j)% 256 - 128); // -128 ~ 127
        }
    }

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            // Wk_b[i][j] = (int32_t)(rand() % 256 - 128); // -128 ~ 127
            Wk_b[i][j] = (int32_t)((i + j)% 256 - 128); // -128 ~ 127
        }
    }

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            // Wv_b[i][j] = (int32_t)(rand() % 256 - 128); // -128 ~ 127
            Wv_b[i][j] = (int32_t)((i + j)% 256 - 128); // -128 ~ 127
        }
    }

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            // Q_in[i][j] = (int8_t)(rand() % 256 - 128); // -128 ~ 127
            Q_in[i][j] = (int8_t)((i + j)% 256 - 128); // -128 ~ 127
            // Q_in[i][j] = (int8_t)(j/64 + 1);
        }
    }

    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            // KT_in[i][j] = (int8_t)(rand() % 256 - 128); // -128 ~ 127
            KT_in[i][j] = (int8_t)((i + j)% 256 - 128); // -128 ~ 127
        }
    }
}

void reorder_A() {
    // int DIM_I = 8, DIM_K = 32;
    int x = 0, y = 0;
    int cnt = 0;
    while(x < SEQ_LEN && y < HIDDEN_DIM) {
        for(int i = 0; i < DIM_I; i++) {
            for(int j = 0; j < DIM_K; j++) {
                inputR[cnt / HIDDEN_DIM][cnt % HIDDEN_DIM] = input[x + i][y + j];
                cnt++;
            }
        }
        if(y + DIM_K >= HIDDEN_DIM) {
            x += DIM_I, y = 0; 
        }
        else {
            y += DIM_K;
        }
    }

}

void reorder_Q_in() {
    const int hd = HIDDEN_DIM / NUM_HEADS;

    for (int head = 0; head < NUM_HEADS; head++) {
        int x = 0, y = 0;
        int cnt = 0;

        // 每个 head 处理一个 [SEQ_LEN][hd] 子矩阵
        while (x < SEQ_LEN && y < hd) {
            for (int i = 0; i < DIM_I; i++) {
                for (int j = 0; j < DIM_K; j++) {
                    int src_row = x + i;
                    int src_col = head * hd + (y + j);

                    int dst_row = cnt / hd;
                    int dst_col = cnt % hd;

                    Q_inR[dst_row][head * hd + dst_col] =
                        Q_in[src_row][src_col];

                    cnt++;
                }
            }

            if (y + DIM_K >= hd) {
                x += DIM_I;
                y = 0;
            } else {
                y += DIM_K;
            }
        }
    }
}

static inline void qkv_gemm_golden_int32(
    const elem_t* A,      // [SEQ_LEN][HIDDEN_DIM]
    const elem_t* B,      // [HIDDEN_DIM][HIDDEN_DIM]
    const acc_t*  D,      // [SEQ_LEN][HIDDEN_DIM] or NULL
    acc_t*        C,      // [SEQ_LEN][HIDDEN_DIM]
    size_t M,             // SEQ_LEN
    size_t N,             // HIDDEN_DIM
    size_t K              // HIDDEN_DIM
)
{
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            acc_t sum = D ? D[i * N + j] : 0;
            for (size_t k = 0; k < K; k++) {
                sum += (acc_t)A[i * K + k] *
                       (acc_t)B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static inline void qkv_golden(
    int seq_len,
    int hidden_dim,

    const elem_t* input,
    const elem_t* enc_out,

    const elem_t* Wq,
    const elem_t* Wk,
    const elem_t* Wv,

    const acc_t* Wq_b,
    const acc_t* Wk_b,
    const acc_t* Wv_b,

    acc_t* Q_buf,
    acc_t* K_buf,
    acc_t* V_buf
)
{
    // Q = input * Wq + Wq_b
    qkv_gemm_golden_int32(
        input, Wq, Wq_b, Q_buf,
        seq_len, hidden_dim, hidden_dim
    );

    // K = enc_out * Wk + Wk_b
    qkv_gemm_golden_int32(
        input, Wk, Wk_b, K_buf,
        seq_len, hidden_dim, hidden_dim
    );

    // V = enc_out * Wv + Wv_b
    qkv_gemm_golden_int32(
        input, Wv, Wv_b, V_buf,
        seq_len, hidden_dim, hidden_dim
    );
}


static inline void attn_gemm_head_golden_int32(
    const elem_t *A,      // Q_in + head*hd
    const elem_t *B,      // KT_in + head*hd*seq_len
    acc_t *C,            // attn_buf + head*seq_len*seq_len
    int seq_len,
    int hidden_dim,
    int hd
)
{
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            acc_t sum = 0;
            for (int k = 0; k < hd; k++) {
                sum +=
                    A[i * hidden_dim + k] *   // Q[i][head*hd + k]
                    B[k * seq_len + j];       // KT[head*hd + k][j]
            }
            C[i * seq_len + j] = sum;
        }
    }
}

void attn_gemm_golden_int32(
    const elem_t *Q_in,     // [SEQ_LEN][HIDDEN_DIM]
    const elem_t *KT_in,    // [HIDDEN_DIM][SEQ_LEN]
    acc_t *attn_g,         // [NUM_HEADS][SEQ_LEN][SEQ_LEN]
    int seq_len,
    int hidden_dim,
    int num_heads
)
{
    int hd = hidden_dim / num_heads;

    for (int head = 0; head < num_heads; head++) {
        const elem_t *A = Q_in  + head * hd;
        const elem_t *B = KT_in + head * hd * seq_len;
        acc_t *C = attn_g + head * seq_len * seq_len;

        attn_gemm_head_golden_int32(
            A, B, C,
            seq_len,
            hidden_dim, 
            hd
        );
    }
}


// 比较两个int32矩阵是否完全相等
static inline bool matcmp_int32(const int32_t* ref, const int32_t* test, size_t D_M, size_t D_N)
{
    for (size_t i = 0; i < D_M * D_N; i++) {
        if (ref[i] != test[i]) {
            return false;
        }
    }
    return true;
}

static inline bool attn_cmp_int32(
    const acc_t* ref,
    const acc_t* test,
    int num_heads,
    int seq_len
)
{
    size_t total = (size_t)num_heads * seq_len * seq_len;
    for (size_t i = 0; i < total; i++) {
        if (ref[i] != test[i]) {
            return false;
        }
    }
    return true;
}

static inline void dump_matrix_int8(
    FILE *fp, const char *name,
    const int8_t *mat, size_t M, size_t N)
{
    fprintf(fp, "%s [%zu x %zu]:\n", name, M, N);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            fprintf(fp, "%d ", mat[i * N + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
}

static inline void dump_matrix_int32(
    FILE *fp, const char *name,
    const int32_t *mat, size_t M, size_t N)
{
    fprintf(fp, "%s [%zu x %zu]:\n", name, M, N);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            fprintf(fp, "%d ", mat[i * N + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
}

static inline void dump_qkv_report(
    const char *path,

    const elem_t *input,
    const elem_t *Wq, const elem_t *Wk, const elem_t *Wv,
    const acc_t  *Wq_b, const acc_t  *Wk_b, const acc_t  *Wv_b,

    const acc_t *Q_buf,   const acc_t *K_buf,   const acc_t *V_buf,
    const acc_t *Q_buf_g, const acc_t *K_buf_g, const acc_t *V_buf_g
)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("open QKV_cp.out failed");
        return;
    }

    /* ================= 输入矩阵 ================= */
    // fprintf(fp, "================ INPUT MATRICES ================\n");
    // dump_matrix_int8(fp,  "input", input, SEQ_LEN, HIDDEN_DIM);
    // dump_matrix_int8(fp,  "Wq",    Wq,    HIDDEN_DIM, HIDDEN_DIM);
    // dump_matrix_int8(fp,  "Wk",    Wk,    HIDDEN_DIM, HIDDEN_DIM);
    // dump_matrix_int8(fp,  "Wv",    Wv,    HIDDEN_DIM, HIDDEN_DIM);
    // dump_matrix_int32(fp, "Wq_b",  Wq_b,  SEQ_LEN, HIDDEN_DIM);
    // dump_matrix_int32(fp, "Wk_b",  Wk_b,  SEQ_LEN, HIDDEN_DIM);
    // dump_matrix_int32(fp, "Wv_b",  Wv_b,  SEQ_LEN, HIDDEN_DIM);

    /* ================= 实际输出 ================= */
    fprintf(fp, "================ HW OUTPUT MATRICES ================\n");
    dump_matrix_int32(fp, "Q_buf", Q_buf, SEQ_LEN, HIDDEN_DIM);
    dump_matrix_int32(fp, "K_buf", K_buf, SEQ_LEN, HIDDEN_DIM);
    dump_matrix_int32(fp, "V_buf", V_buf, SEQ_LEN, HIDDEN_DIM);

    /* ================= golden 输出 ================= */
    // fprintf(fp, "================ GOLDEN OUTPUT MATRICES ================\n");
    // dump_matrix_int32(fp, "Q_buf_g", Q_buf_g, SEQ_LEN, HIDDEN_DIM);
    // dump_matrix_int32(fp, "K_buf_g", K_buf_g, SEQ_LEN, HIDDEN_DIM);
    // dump_matrix_int32(fp, "V_buf_g", V_buf_g, SEQ_LEN, HIDDEN_DIM);

    /* ================= 对拍结果 ================= */
    // fprintf(fp, "================ COMPARE RESULT ================\n");

    // fprintf(fp, "Q : %s\n",
    //     matcmp_int32(Q_buf, Q_buf_g, SEQ_LEN, HIDDEN_DIM) ? "PASS" : "FAIL");

    // fprintf(fp, "K : %s\n",
    //     matcmp_int32(K_buf, K_buf_g, SEQ_LEN, HIDDEN_DIM) ? "PASS" : "FAIL");

    // fprintf(fp, "V : %s\n",
    //     matcmp_int32(V_buf, V_buf_g, SEQ_LEN, HIDDEN_DIM) ? "PASS" : "FAIL");

    fclose(fp);
}

static inline void dump_attn_report(
    const char *path,

    const elem_t *Q_in,
    const elem_t *KT_in,

    const acc_t *attn_buf,     // HW output
    const acc_t *attn_buf_g    // golden output
)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("open attn_cp.out failed");
        return;
    }

    /* ================= 输入矩阵 ================= */
    // fprintf(fp, "================ INPUT MATRICES ================\n");
    // dump_matrix_int8(fp, "Q_in", Q_in, SEQ_LEN, HIDDEN_DIM);
    // dump_matrix_int8(fp, "Q_inR", Q_inR, SEQ_LEN, HIDDEN_DIM);

    // dump_matrix_int8(fp, "KT_in", KT_in, SEQ_LEN, HIDDEN_DIM);

    /* ================= 实际输出 ================= */
    fprintf(fp, "================ HW OUTPUT MATRICES ================\n");
    for (int h = 0; h < NUM_HEADS; h++) {
        char name[64];
        snprintf(name, sizeof(name), "attn_buf_head_%d", h);
        dump_matrix_int32(
            fp,
            name,
            attn_buf + h * SEQ_LEN * SEQ_LEN,
            SEQ_LEN,
            SEQ_LEN
        );
    }

    /* ================= golden 输出 ================= */
    // fprintf(fp, "================ GOLDEN OUTPUT MATRICES ================\n");
    // for (int h = 0; h < NUM_HEADS; h++) {
    //     char name[64];
    //     snprintf(name, sizeof(name), "attn_buf_g_head_%d", h);
    //     dump_matrix_int32(
    //         fp,
    //         name,
    //         attn_buf_g + h * SEQ_LEN * SEQ_LEN,
    //         SEQ_LEN,
    //         SEQ_LEN
    //     );
    // }

    /* ================= 对拍结果 ================= */
    // fprintf(fp, "================ COMPARE RESULT ================\n");
    // for (int h = 0; h < NUM_HEADS; h++) {
    //     bool pass = matcmp_int32(
    //         attn_buf   + h * SEQ_LEN * SEQ_LEN,
    //         attn_buf_g + h * SEQ_LEN * SEQ_LEN,
    //         SEQ_LEN,
    //         SEQ_LEN
    //     );
    //     fprintf(fp, "HEAD %d : %s\n", h, pass ? "PASS" : "FAIL");
    // }

    fclose(fp);
}

/* ---------------------------------------------------  */
// matmul test end
/* ---------------------------------------------------  */

/* ---------------------------------------------------  */
// softmax test begin
/* ---------------------------------------------------  */





int main() {
    printf("start test!\n");
    printf("Attention test start...\n");
    printf("SEQ_LEN=%d, HIDDEN_DIM=%d, NUM_HEADS=%d\n", SEQ_LEN, HIDDEN_DIM, NUM_HEADS);
    
    // rand_init_matmul();
    // reorder_A();
    // reorder_Q_in();
    // load_C_from_file("/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/0120/number_quantized.txt");   
    
    attention(HIDDEN_DIM, NUM_HEADS, SEQ_LEN,
        (const elem_t *)inputR, (const elem_t *)inputR,
        (const elem_t *)Wq, (const elem_t *)Wk, (const elem_t *)Wv,
        (const acc_t *)Wq_b, (const acc_t *)Wk_b, (const acc_t *)Wv_b,
        (acc_t *)Q_buf, (acc_t *)K_buf, (acc_t *)V_buf,
        (const elem_t *)Q_inR, (const elem_t *)KT_in,
        (acc_t *)attn_buf, 
        (const acc_t *)softmax_in, (acc_t *)softmax_attn_buf,
        (const elem_t *)attn_in, (const elem_t *)V_in,
        (acc_t *)out_buf,
        (const elem_t *)out_in, (const elem_t *)Wo, (const acc_t *)Wo_b,
        (acc_t *)out_buf_acc,
        (const acc_t *)resadd_input,
        (acc_t *)resadd_out, 
        (const acc_t *)LN_in, (acc_t *)out
    );
    printf("Attention test end.\n");
    printf("end test!\n");

    // /* ---------------------------------------------------------- */
    // // QKV_matmul_test
    // // golden
    // qkv_golden(
    //     SEQ_LEN, HIDDEN_DIM,
    //     input, input,   // enc_out == input
    //     Wq, Wk, Wv,
    //     Wq_b, Wk_b, Wv_b,
    //     Q_buf_g, K_buf_g, V_buf_g
    // );

    // if (matcmp_int32(Q_buf, Q_buf_g, SEQ_LEN, HIDDEN_DIM)) {
    //     printf("Q PASS!!\n");
    // } else {
    //     printf("Q FAIL!!\n");
    // }

    // if (matcmp_int32(K_buf, K_buf_g, SEQ_LEN, HIDDEN_DIM)) {
    //     printf("K PASS!!\n");
    // } else {
    //     printf("K FAIL!!\n");
    // }

    // if (matcmp_int32(V_buf, V_buf_g, SEQ_LEN, HIDDEN_DIM)) {
    //     printf("V PASS!!\n");
    // } else {
    //     printf("V FAIL!!\n");
    // }

    dump_qkv_report(
        "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/0120/QKV_cp.out",

        (const elem_t *)input,
        (const elem_t *)Wq,
        (const elem_t *)Wk,
        (const elem_t *)Wv,
        (const acc_t  *)Wq_b,
        (const acc_t  *)Wk_b,
        (const acc_t  *)Wv_b,

        (const acc_t *)Q_buf,
        (const acc_t *)K_buf,
        (const acc_t *)V_buf,

        (const acc_t *)Q_buf_g,
        (const acc_t *)K_buf_g,
        (const acc_t *)V_buf_g
    );

    // /* ---------------------------------------------------------- */

    /* ---------------------------------------------------------- */
    // attn_matmul_test
    // attn_gemm_golden_int32(
    //     (const elem_t*)Q_in,
    //     (const elem_t*)KT_in,
    //     (acc_t*)attn_buf_g,
    //     SEQ_LEN,
    //     HIDDEN_DIM,
    //     NUM_HEADS
    // );

    // dump_attn_report(
    //     "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/0120/attn_cp.out",
    //     (const elem_t*)Q_in,
    //     (const elem_t*)KT_in,
    //     (const acc_t*)attn_buf,
    //     (const acc_t*)attn_buf_g
    // );
    /* ---------------------------------------------------------- */



    return 0;
}



