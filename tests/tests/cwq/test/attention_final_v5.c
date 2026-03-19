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
#define EXPANSION_DIM (HIDDEN_DIM)

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
static alignas(CACHELINE_SIZE) acc_t resadd_input0[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
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
static alignas(CACHELINE_SIZE) acc_t softmax_attn_buf_g[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {[0 ... NUM_HEADS-1][0 ... SEQ_LEN-1][0 ... SEQ_LEN-1] = 1};
static alignas(CACHELINE_SIZE) acc_t resadd_out_g[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t out_g[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};

static alignas(CACHELINE_SIZE) elem_t inputR[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t Q_inR[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t Q_inR_pad[SEQ_LEN][4*HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... 4*HIDDEN_DIM-1] = 1};

// ffn
static alignas(CACHELINE_SIZE) elem_t ffn_input[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t ffn_ff1_w[HIDDEN_DIM][EXPANSION_DIM] = {[0 ... HIDDEN_DIM-1][0 ... EXPANSION_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t ffn_ff2_w[EXPANSION_DIM][HIDDEN_DIM] = {[0 ... EXPANSION_DIM-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ffn_ff1_b[SEQ_LEN][EXPANSION_DIM] = {[0 ... SEQ_LEN-1][0 ... EXPANSION_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ffn_ff2_b[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ffn_ff1_out[SEQ_LEN][EXPANSION_DIM] = {[0 ... SEQ_LEN-1][0 ... EXPANSION_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ffn_ff2_out[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ffn_gelu_out[SEQ_LEN][EXPANSION_DIM] = {[0 ... SEQ_LEN-1][0 ... EXPANSION_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ffn_ln_out[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ffn_out[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ffn_resadd_input[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ffn_ff2_in[SEQ_LEN][EXPANSION_DIM] = {[0 ... SEQ_LEN-1][0 ... EXPANSION_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ffn_ln_in[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 17};
static alignas(CACHELINE_SIZE) acc_t ffn_gelu_in[SEQ_LEN][EXPANSION_DIM] = {[0 ... SEQ_LEN-1][0 ... EXPANSION_DIM-1] = 1};

static alignas(CACHELINE_SIZE) acc_t ffn_gelu_out_g[SEQ_LEN][EXPANSION_DIM] = {[0 ... SEQ_LEN-1][0 ... EXPANSION_DIM-1] = 1};



// layernorm real input

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

        const acc_t * resadd_input0,
        const acc_t * resadd_input,
        acc_t * resadd_out, 
        const acc_t * LN_in,
        acc_t * out
)
{
    int hidden_dim_per_head = hidden_dim / num_heads; // 64

    // // Q = Wq * input + Wq_b
    // // K = Wk * enc_out + Wk_b
    // // V = Wv * enc_out + Wv_b
    // const int qkv_matmuls_n = 3;
    // for (int i = 0; i < qkv_matmuls_n; i++) {
    //     const elem_t * qkv_weights[] = {Wq, Wk, Wv};
    //     const elem_t * qkv_ins[] = {input, enc_out, enc_out};
    //     const acc_t * qkv_bs[] = {Wq_b, Wk_b, Wv_b};
    //     acc_t * qkv_outs[] = {Q_buf, K_buf, V_buf};
    //     const elem_t * qkv_w = qkv_weights[i];
    //     const elem_t * qkv_in = qkv_ins[i];
    //     const acc_t * qkv_b = qkv_bs[i];
    //     acc_t * qkv_out = qkv_outs[i];
    //     tiled_matmul_auto(seq_len, hidden_dim, hidden_dim, qkv_in, 
    //         qkv_w, qkv_b, qkv_out, hidden_dim, hidden_dim, hidden_dim, hidden_dim);
    // }
    // printf("QKV matmuls done.\n");


    // // attn = Q * K^T
    // for (int head = 0; head < num_heads; head++) {
    //     const elem_t * A = Q_in + head * 256 /* hidden_dim_per_head */;
    //     const elem_t * B = KT_in + head * hidden_dim_per_head * seq_len;
    //     acc_t * C = attn_buf + head * seq_len * seq_len;
    //     tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head, A, B, NULL, C,
    //         4 * hidden_dim, seq_len, 0, seq_len);
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
    //     resadd_input0, resadd_input, resadd_out, hidden_dim, hidden_dim, hidden_dim);
    // // printf("resadd_out printed\n");

    // for (int i = 0; i < SEQ_LEN; i++) {
    //     for(int j = 0; j < HIDDEN_DIM; j++) {
    //         printf("%d ", resadd_out[i * HIDDEN_DIM + j]);
    //     }
    //     printf("\n");
    // }

    // // out = LN(resadd_out)
    tiled_norm_auto(seq_len, hidden_dim, LN_in, out,
            1.0 / (1<<0), 0, hidden_dim, hidden_dim, LAYERNORM);
    // printf("LayerNorm done.\n");
}

/* ---------------------------------------------------  */
// matmul test begin
/* ---------------------------------------------------  */

void rand_init_matmul() {
    // srand((unsigned)time(NULL));
    // for (int i = 0; i < SEQ_LEN; i++) {
    //     for (int j = 0; j < HIDDEN_DIM; j++) {
    //         // input[i][j] = (int8_t)(rand() % 256 - 128); // -128 ~ 127
    //         input[i][j] = (int8_t)((i + j)% 256 - 128); // -128 ~ 127
    //     }
    // }

    // for (int i = 0; i < HIDDEN_DIM; i++) {
    //     for (int j = 0; j < HIDDEN_DIM; j++) {
    //         // Wq[i][j] = (int8_t)(rand() % 256 - 128); // -128 ~ 127
    //         Wq[i][j] = (int8_t)((i + j)% 256 - 128); // -128 ~ 127
    //         // Wq[i][j] = (int8_t)(i % 128); // -128 ~ 127
    //     }
    // }

    // for (int i = 0; i < HIDDEN_DIM; i++) {
    //     for (int j = 0; j < HIDDEN_DIM; j++) {
    //         // Wk[i][j] = (int8_t)(rand() % 256 - 128); // -128 ~ 127
    //         Wk[i][j] = (int8_t)((i + j)% 256 - 128); // -128 ~ 127
    //     }
    // }

    // for (int i = 0; i < HIDDEN_DIM; i++) {
    //     for (int j = 0; j < HIDDEN_DIM; j++) {
    //         // Wv[i][j] = (int8_t)(rand() % 256 - 128); // -128 ~ 127
    //         Wv[i][j] = (int8_t)((i + j)% 256 - 128); // -128 ~ 127
    //     }
    // }

    // for (int i = 0; i < SEQ_LEN; i++) {
    //     for (int j = 0; j < HIDDEN_DIM; j++) {
    //         // Wq_b[i][j] = (int32_t)(rand() % 256 - 128); // -128 ~ 127
    //         Wq_b[i][j] = (int32_t)((i + j)% 256 - 128); // -128 ~ 127
    //     }
    // }

    // for (int i = 0; i < SEQ_LEN; i++) {
    //     for (int j = 0; j < HIDDEN_DIM; j++) {
    //         // Wk_b[i][j] = (int32_t)(rand() % 256 - 128); // -128 ~ 127
    //         Wk_b[i][j] = (int32_t)((i + j)% 256 - 128); // -128 ~ 127
    //     }
    // }

    // for (int i = 0; i < SEQ_LEN; i++) {
    //     for (int j = 0; j < HIDDEN_DIM; j++) {
    //         // Wv_b[i][j] = (int32_t)(rand() % 256 - 128); // -128 ~ 127
    //         Wv_b[i][j] = (int32_t)((i + j)% 256 - 128); // -128 ~ 127
    //     }
    // }

    // for (int i = 0; i < SEQ_LEN; i++) {
    //     for (int j = 0; j < HIDDEN_DIM; j++) {
    //         // Q_in[i][j] = (int8_t)(rand() % 256 - 128); // -128 ~ 127
    //         Q_in[i][j] = (int8_t)((i + j)% 256 - 128); // -128 ~ 127
    //         // Q_in[i][j] = (int8_t)(j/64 + 1);
    //     }
    // }

    // for (int i = 0; i < HIDDEN_DIM; i++) {
    //     for (int j = 0; j < SEQ_LEN; j++) {
    //         // KT_in[i][j] = (int8_t)(rand() % 256 - 128); // -128 ~ 127
    //         KT_in[i][j] = (int8_t)((i + j)% 256 - 128); // -128 ~ 127
    //     }
    // }

    // for (int num = 0; num < NUM_HEADS; num++) {
    //     for (int i = 0; i < SEQ_LEN; i++) {
    //         for (int j = 0; j < SEQ_LEN; j++) {
    //             // softmax_in[num][i][j] = (int32_t)(rand() % 65536 - 32768); 
    //             softmax_in[num][i][j] = (int32_t)((i + 10 * j + 100 * num)% 65536); 
    //         }
    //     }
    // }

    // for (int i = 0; i < SEQ_LEN; i++) {
    //     for (int j = 0; j < HIDDEN_DIM; j++) {
    //         resadd_input0[i][j] = (int32_t)((i + j)% 256 - 128); // -128 ~ 127
    //         resadd_input[i][j] = (int32_t)((i + j)% 256 - 128); // -128 ~ 127
    //     }
    // }
    
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            LN_in[i][j] = (int32_t)((i + j)% 8 - 4); // -128 ~ 127
        }
    }
    
    // for (int i = 0; i < SEQ_LEN; i++) {
    //     for (int j = 0; j < EXPANSION_DIM; j++) {
    //         ffn_gelu_in[i][j] = (int32_t)(rand()% 256 - 128); // -128 ~ 127
    //     }
    // }
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

void pad_Q_inR_to_256()
{
    const int hd = HIDDEN_DIM / NUM_HEADS;   // 原始 head dim
    const int PAD_HD = 256;                  // 目标 head dim

    for (int head = 0; head < NUM_HEADS; head++) {
        int src_col_base = head * hd;
        int dst_col_base = head * PAD_HD;

        for (int i = 0; i < SEQ_LEN; i++) {

            /* 1. 拷贝有效数据 */
            for (int j = 0; j < hd; j++) {
                Q_inR_pad[i][dst_col_base + j] =
                    Q_inR[i][src_col_base + j];
            }

            /* 2. 右侧 padding 置 0 */
            for (int j = hd; j < PAD_HD; j++) {
                Q_inR_pad[i][dst_col_base + j] = 0;
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
    fprintf(fp, "================ INPUT MATRICES ================\n");
    dump_matrix_int8(fp, "Q_in", Q_in, SEQ_LEN, HIDDEN_DIM);
    dump_matrix_int8(fp, "Q_inR_pad", Q_inR_pad, SEQ_LEN, 4 * HIDDEN_DIM);

    dump_matrix_int8(fp, "KT_in", KT_in, SEQ_LEN, HIDDEN_DIM);

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
    fprintf(fp, "================ GOLDEN OUTPUT MATRICES ================\n");
    for (int h = 0; h < NUM_HEADS; h++) {
        char name[64];
        snprintf(name, sizeof(name), "attn_buf_g_head_%d", h);
        dump_matrix_int32(
            fp,
            name,
            attn_buf_g + h * SEQ_LEN * SEQ_LEN,
            SEQ_LEN,
            SEQ_LEN
        );
    }

    /* ================= 对拍结果 ================= */
    fprintf(fp, "================ COMPARE RESULT ================\n");
    for (int h = 0; h < NUM_HEADS; h++) {
        bool pass = matcmp_int32(
            attn_buf   + h * SEQ_LEN * SEQ_LEN,
            attn_buf_g + h * SEQ_LEN * SEQ_LEN,
            SEQ_LEN,
            SEQ_LEN
        );
        fprintf(fp, "HEAD %d : %s\n", h, pass ? "PASS" : "FAIL");
    }

    fclose(fp);
}


/* ---------------------------------------------------  */
// matmul test end
/* ---------------------------------------------------  */

/* ---------------------------------------------------  */
// softmax test begin
/* ---------------------------------------------------  */

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
        printf("Row %lu: \n", i);
        for (size_t j = 0; j < D_N; j++) {
            printf("%f ", vals[j]);
        }
        printf("\n");
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
        for (size_t j = 0; j < D_N; j++) {
            printf("%f ", vals[j]);
        }
        printf("\n");
        // normalize
        for (size_t j = 0; j < D_N; j++) {
            vals[j] = vals[j] / sum_exp;
            C_G[i * D_N + j] = (int32_t)(vals[j] * output_scale_factor);
        }
        for (size_t j = 0; j < D_N; j++) {
            printf("%f ", vals[j]);
        }
        printf("\n");
        for (size_t j = 0; j < D_N; j++) {
            printf("%d ", C_G[i * D_N + j]);
        }
        printf("\n");
    }
}

void softmax_golden_all_heads(
    const acc_t *softmax_in,          // [NUM_HEADS][SEQ_LEN][SEQ_LEN]
    acc_t *softmax_attn_buf_g,        // [NUM_HEADS][SEQ_LEN][SEQ_LEN]
    int num_heads,
    int seq_len,
    int input_q_bits,
    int output_q_bits
)
{
    for (int h = 0; h < num_heads; h++) {
        const acc_t *C =
            softmax_in + h * seq_len * seq_len;
        acc_t *C_G =
            softmax_attn_buf_g + h * seq_len * seq_len;

        softmax_golden_int32(
            C,
            C_G,
            seq_len,
            seq_len,
            input_q_bits,
            output_q_bits
        );
    }
}

double row_mse(const int32_t *a, const int32_t *b, size_t D_N) {
    double sum_sq = 0.0;
    for (size_t j = 0; j < D_N; j++) {
        double diff = (double)a[j] - (double)b[j];
        sum_sq += diff * diff;
    }
    return sum_sq / D_N;
}

double softmax_head_mse(
    const acc_t *T,   // HW output [SEQ_LEN][SEQ_LEN]
    const acc_t *G,   // golden    [SEQ_LEN][SEQ_LEN]
    int seq_len
)
{
    double total_mse = 0.0;

    for (int i = 0; i < seq_len; i++) {
        const acc_t *row_T = T + i * seq_len;
        const acc_t *row_G = G + i * seq_len;
        total_mse += row_mse(row_T, row_G, seq_len);
    }

    return total_mse / seq_len;
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

double softmax_all_heads_mse(
    const acc_t *softmax_attn_buf,     // HW
    const acc_t *softmax_attn_buf_g,   // golden
    int num_heads,
    int seq_len
)
{
    double total_mse = 0.0;

    for (int h = 0; h < num_heads; h++) {
        const acc_t *T =
            softmax_attn_buf   + h * seq_len * seq_len;
        const acc_t *G =
            softmax_attn_buf_g + h * seq_len * seq_len;

        total_mse += softmax_head_mse(T, G, seq_len);
    }

    return total_mse / num_heads;
}

double softmax_all_heads_cos(
    const acc_t *softmax_attn_buf,     // HW
    const acc_t *softmax_attn_buf_g,   // golden
    int num_heads,
    int seq_len
)
{
    double total_cos = 0.0;

    for (int h = 0; h < num_heads; h++) {
        const acc_t *T =
            softmax_attn_buf   + h * seq_len * seq_len;
        const acc_t *G =
            softmax_attn_buf_g + h * seq_len * seq_len;

        total_cos += row_cosine_similarity(T, G, seq_len);  
    }

    return total_cos / num_heads;
}

// KL散度

double row_kl_divergence(
    const int32_t *P_q,   // golden
    const int32_t *Q_q,   // hw
    size_t D_N,
    double scale          // = (double)(1 << output_q_bits)
)
{
    const double eps = 1e-12;
    double kl = 0.0;

    for (size_t j = 0; j < D_N; j++) {
        double p = (double)P_q[j] / scale;
        double q = (double)Q_q[j] / scale;

        if (p < eps) continue;          // p==0 → 贡献为 0
        if (q < eps) q = eps;

        kl += p * log(p / q);
    }
    return kl;
}

double softmax_head_kl(
    const acc_t *T,   // HW output
    const acc_t *G,   // golden
    int seq_len,
    int output_q_bits
)
{
    double scale = (double)(1 << output_q_bits);
    double total_kl = 0.0;

    for (int i = 0; i < seq_len; i++) {
        total_kl += row_kl_divergence(
            G + i * seq_len,
            T + i * seq_len,
            seq_len,
            scale
        );
    }

    return total_kl / seq_len;
}

double softmax_all_heads_kl(
    const acc_t *softmax_attn_buf,     // HW
    const acc_t *softmax_attn_buf_g,   // golden
    int num_heads,
    int seq_len,
    int output_q_bits
)
{
    double total_kl = 0.0;

    for (int h = 0; h < num_heads; h++) {
        const acc_t *T =
            softmax_attn_buf   + h * seq_len * seq_len;
        const acc_t *G =
            softmax_attn_buf_g + h * seq_len * seq_len;

        total_kl += softmax_head_kl(
            T, G, seq_len, output_q_bits
        );
    }

    return total_kl / num_heads;
}


void dump_softmax_report(
    const char *path,
    const acc_t *softmax_in,
    const acc_t *softmax_attn_buf,
    const acc_t *softmax_attn_buf_g,
    int num_heads,
    int seq_len,
    int output_q_bits
)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("open softmax_cp.out failed");
        return;
    }

    fprintf(fp, "================ INPUT MATRICES ================\n");
    for (int h = 0; h < num_heads; h++) {
        char name[64];
        snprintf(name, sizeof(name), "softmax_in_head_%d", h);
        dump_matrix_int32(
            fp,
            name,
            softmax_in + h * seq_len * seq_len,
            seq_len,
            seq_len
        );
    }

    fprintf(fp, "================ HW OUTPUT MATRICES ================\n");
    for (int h = 0; h < num_heads; h++) {
        char name[64];
        snprintf(name, sizeof(name), "softmax_out_head_%d", h);
        dump_matrix_int32(
            fp,
            name,
            softmax_attn_buf + h * seq_len * seq_len,
            seq_len,
            seq_len
        );
    }

    fprintf(fp, "================ GOLDEN OUTPUT MATRICES ================\n");
    for (int h = 0; h < num_heads; h++) {
        char name[64];
        snprintf(name, sizeof(name), "softmax_golden_head_%d", h);
        dump_matrix_int32(
            fp,
            name,
            softmax_attn_buf_g + h * seq_len * seq_len,
            seq_len,
            seq_len
        );
    }

    fprintf(fp, "================ EVALUATION REPORT ================\n");

    double global_mse = 0.0;
    double global_cos = 0.0;
    double global_kl  = 0.0;

    for (int h = 0; h < num_heads; h++) {
        double mse = softmax_head_mse(
            softmax_attn_buf   + h * seq_len * seq_len,
            softmax_attn_buf_g + h * seq_len * seq_len,
            seq_len
        );

        double cos = row_cosine_similarity(
            softmax_attn_buf   + h * seq_len * seq_len,
            softmax_attn_buf_g + h * seq_len * seq_len,
            seq_len
        );

        double kl = softmax_head_kl(
            softmax_attn_buf   + h * seq_len * seq_len,
            softmax_attn_buf_g + h * seq_len * seq_len,
            seq_len,
            output_q_bits
        );

        fprintf(fp, "\nHEAD %d MSE = %.6e\n", h, mse);
        fprintf(fp, "HEAD %d COS = %.6e\n", h, cos);
        fprintf(fp, "HEAD %d KL  = %.6e\n", h, kl);

        global_mse += mse;
        global_cos += cos;
        global_kl  += kl;
    }

    fprintf(fp, "\nAVERAGE MSE (ALL HEADS) = %.6e\n", global_mse / num_heads);
    fprintf(fp, "AVERAGE COS (ALL HEADS) = %.6e\n", global_cos / num_heads);
    fprintf(fp, "AVERAGE KL  (ALL HEADS) = %.6e\n", global_kl  / num_heads);


    fclose(fp);
}

/* ---------------------------------------------------  */
// softmax test end
/* ---------------------------------------------------  */

/* ---------------------------------------------------  */
// resadd test begin
/* ---------------------------------------------------  */

static inline void resadd_golden_int32(
    const acc_t *A,    // resadd_input0 [SEQ_LEN][HIDDEN_DIM]
    const acc_t *B,    // resadd_input  [SEQ_LEN][HIDDEN_DIM]
    acc_t *C,          // resadd_out_g  [SEQ_LEN][HIDDEN_DIM]
    int M,
    int N
)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = A[i * N + j] + B[i * N + j];
        }
    }
}

static inline void dump_resadd_report(
    const char *path,

    const acc_t *resadd_input0,
    const acc_t *resadd_input,

    const acc_t *resadd_out,     // HW
    const acc_t *resadd_out_g    // golden
)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("open resadd_cp.out failed");
        return;
    }

    /* ================= 输入矩阵 ================= */
    fprintf(fp, "================ INPUT MATRICES ================\n");
    dump_matrix_int32(fp, "resadd_input0", resadd_input0, SEQ_LEN, HIDDEN_DIM);
    dump_matrix_int32(fp, "resadd_input ", resadd_input,  SEQ_LEN, HIDDEN_DIM);

    /* ================= HW 输出 ================= */
    fprintf(fp, "================ HW OUTPUT MATRIX ================\n");
    dump_matrix_int32(fp, "resadd_out", resadd_out, SEQ_LEN, HIDDEN_DIM);

    /* ================= golden 输出 ================= */
    fprintf(fp, "================ GOLDEN OUTPUT MATRIX ================\n");
    dump_matrix_int32(fp, "resadd_out_g", resadd_out_g, SEQ_LEN, HIDDEN_DIM);

    /* ================= 对拍结果 ================= */
    fprintf(fp, "================ COMPARE RESULT ================\n");

    bool pass = matcmp_int32(
        resadd_out,
        resadd_out_g,
        SEQ_LEN,
        HIDDEN_DIM
    );

    fprintf(fp, "RESADD : %s\n", pass ? "PASS" : "FAIL");

    fclose(fp);
}


/* ---------------------------------------------------  */
// resadd test end
/* ---------------------------------------------------  */

/* ---------------------------------------------------  */
// layernorm test begin
/* ---------------------------------------------------  */

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


// 将 Q8.24 定点数转换为浮点数
static inline double q824_to_float(int32_t q824_val) {
    return (double)q824_val / 16777216.0;  // 2^24 = 16777216
}

// 完整版 LayerNorm 报告 - 输出全部矩阵
void dump_layernorm_report_full(
    const char *path,
    const acc_t *LN_input,           // 输入 [SEQ_LEN][HIDDEN_DIM], Q8.24
    const acc_t *LN_output,          // HW 输出 [SEQ_LEN][HIDDEN_DIM], Q8.24  
    const acc_t *LN_output_g,        // Golden 输出 [SEQ_LEN][HIDDEN_DIM], Q8.24
    int seq_len,
    int hidden_dim
) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("open layernorm_cp.out failed");
        return;
    }
    
    /* ================= 输入矩阵 ================= */
    // fprintf(fp, "================ INPUT MATRIX [%d x %d] (Q8.24) =================\n", 
    //        seq_len, hidden_dim);
    // fprintf(fp, "Format: Q8.24 (24 fractional bits), raw integer values\n\n");
    
    // for (int i = 0; i < seq_len; i++) {
    //     fprintf(fp, "Row %4d: ", i);
    //     for (int j = 0; j < hidden_dim; j++) {
    //         fprintf(fp, "%11d ", LN_input[i * hidden_dim + j]);
    //     }
    //     fprintf(fp, "\n");
    // }
    // fprintf(fp, "\n");
    
    /* ================= HW 输出矩阵 ================= */
    fprintf(fp, "================ HW OUTPUT MATRIX [%d x %d] (Q8.24) =================\n", 
           seq_len, hidden_dim);
    fprintf(fp, "Format: Q8.24 (24 fractional bits), raw integer values\n\n");
    
    for (int i = 0; i < seq_len; i++) {
        fprintf(fp, "Row %4d: ", i);
        for (int j = 0; j < hidden_dim; j++) {
            fprintf(fp, "%11d ", LN_output[i * hidden_dim + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    
    /* ================= Golden 输出矩阵 ================= */
    // fprintf(fp, "================ GOLDEN OUTPUT MATRIX [%d x %d] (Q8.24) =================\n", 
    //        seq_len, hidden_dim);
    // fprintf(fp, "Format: Q8.24 (24 fractional bits), raw integer values\n\n");
    
    // for (int i = 0; i < seq_len; i++) {
    //     fprintf(fp, "Row %4d: ", i);
    //     for (int j = 0; j < hidden_dim; j++) {
    //         fprintf(fp, "%11d ", LN_output_g[i * hidden_dim + j]);
    //     }
    //     fprintf(fp, "\n");
    // }
    // fprintf(fp, "\n");
    
    /* ================= 比较结果 ================= */
    // fprintf(fp, "================ COMPARISON RESULTS =================\n");
    // fprintf(fp, "Row-by-row analysis (MSE, Cosine Similarity, KL Divergence)\n\n");
    
    // // 计算每行的指标
    // double total_mse = 0.0;
    // double total_cos = 0.0;
    // double total_kl = 0.0;
    // int rows_with_large_kl = 0;
    // const double kl_threshold = 0.01;  // KL散度警告阈值
    
    // for (int i = 0; i < seq_len; i++) {
    //     const acc_t *row_hw = LN_output + i * hidden_dim;
    //     const acc_t *row_golden = LN_output_g + i * hidden_dim;
        
    //     // 计算 MSE
    //     double row_mse = 0.0;
    //     for (int j = 0; j < hidden_dim; j++) {
    //         double diff = q824_to_float(row_hw[j]) - q824_to_float(row_golden[j]);
    //         row_mse += diff * diff;
    //     }
    //     row_mse /= hidden_dim;
    //     total_mse += row_mse;
        
    //     // 计算余弦相似度
    //     double dot = 0.0, norm_hw = 0.0, norm_golden = 0.0;
    //     for (int j = 0; j < hidden_dim; j++) {
    //         double f_hw = q824_to_float(row_hw[j]);
    //         double f_golden = q824_to_float(row_golden[j]);
    //         dot += f_hw * f_golden;
    //         norm_hw += f_hw * f_hw;
    //         norm_golden += f_golden * f_golden;
    //     }
    //     double row_cos = (norm_hw == 0.0 || norm_golden == 0.0) ? 
    //                     0.0 : dot / (sqrt(norm_hw) * sqrt(norm_golden));
    //     total_cos += row_cos;
        
    //     // 计算 KL 散度
    //     double row_kl = 0.0;
    //     const double eps = 1e-12;
    //     for (int j = 0; j < hidden_dim; j++) {
    //         double p = q824_to_float(row_golden[j]);
    //         double q = q824_to_float(row_hw[j]);
            
    //         // 调整概率值以确保非负
    //         if (p < 0) p = 0;
    //         if (q < 0) q = 0;
            
    //         if (p < eps) continue;
    //         if (q < eps) q = eps;
            
    //         row_kl += p * log(p / q);
    //     }
    //     total_kl += row_kl;
        
    //     // 记录KL散度超过阈值的行
    //     if (row_kl > kl_threshold) {
    //         rows_with_large_kl++;
    //     }
        
    //     // 输出每行的比较结果
    //     fprintf(fp, "Row %4d: MSE=%10.2e, COS=%10.6f, KL=%10.2e", 
    //            i, row_mse, row_cos, row_kl);
        
    //     // 标记有问题的行
    //     if (row_kl > kl_threshold) {
    //         fprintf(fp, "  [KL > %.2e]", kl_threshold);
    //     }
    //     fprintf(fp, "\n");
    // }
    
    // /* ================= 汇总统计 ================= */
    // fprintf(fp, "\n================ SUMMARY =================\n");
    // fprintf(fp, "Total rows compared: %d\n", seq_len);
    // fprintf(fp, "Rows with KL > %.2e: %d (%.1f%%)\n", 
    //        kl_threshold, rows_with_large_kl, 
    //        (double)rows_with_large_kl / seq_len * 100.0);
    // fprintf(fp, "\n");
    
    // double avg_mse = total_mse / seq_len;
    // double avg_cos = total_cos / seq_len;
    // double avg_kl = total_kl / seq_len;
    
    // fprintf(fp, "Average MSE:      %10.2e\n", avg_mse);
    // fprintf(fp, "Average Cosine:   %10.6f\n", avg_cos);
    // fprintf(fp, "Average KL Div:   %10.2e\n", avg_kl);
    
    // /* ================= 评估结论 ================= */
    // fprintf(fp, "\n================ ASSESSMENT =================\n");
    
    // // 设置评估阈值
    // const double mse_threshold = 1e-6;
    // const double cos_threshold = 0.999;
    
    // bool pass_mse = (avg_mse < mse_threshold);
    // bool pass_cos = (avg_cos > cos_threshold);
    // bool pass_kl = (avg_kl < kl_threshold && rows_with_large_kl == 0);
    
    // fprintf(fp, "MSE test (avg < %.1e):     %s\n", 
    //        mse_threshold, pass_mse ? "PASS" : "FAIL");
    // fprintf(fp, "Cosine test (avg > %.3f):  %s\n", 
    //        cos_threshold, pass_cos ? "PASS" : "FAIL");
    // fprintf(fp, "KL test (avg < %.2e & no rows > threshold): %s\n", 
    //        kl_threshold, pass_kl ? "PASS" : "FAIL");
    
    // fprintf(fp, "\nOverall result: ");
    // if (pass_mse && pass_cos && pass_kl) {
    //     fprintf(fp, "PASS\n");
    // } else {
    //     fprintf(fp, "FAIL\n");
    // }
    
    fclose(fp);
    // printf("LayerNorm full report saved to: %s\n", path);
    // printf("  Input matrix:  %d x %d (Q8.24)\n", seq_len, hidden_dim);
    // printf("  Output format: raw integer values\n");
    // printf("  Total values:  %d\n", seq_len * hidden_dim * 3);  // 三个矩阵
}

/* ---------------------------------------------------  */
// layernorm test end
/* ---------------------------------------------------  */






/* ---------------------------------------------------  */
// ffn test begin
/* ---------------------------------------------------  */

void ffn (int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, acc_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,
        const acc_t * resadd_input,
        acc_t * ff2_in,
        acc_t * ff1_out, acc_t * ff2_out,
        acc_t * ln_in,
        const acc_t * gelu_in,
        acc_t * gelu_out, acc_t * ln_out
){

//         // FFN Layer 1: out_buf = GELU(input * ff1_w + ff1_b)
//         tiled_matmul_auto(/*DIM_I=*/seq_len, /*DIM_J=*/expansion_dim, /*DIM_K=*/hidden_dim,
//             /*A=*/ input, /*B=*/ ff1_w,
//             /*D=*/ ff1_b, /*C=*/ ff1_out,
//             /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim);
        
//         printf("MatMul 1 done.\n");
//         for(int i = 0; i < SEQ_LEN; i++) {
//             for(int j = 0; j < EXPANSION_DIM; j++) {
//                 printf("%d ", ff1_out[i * EXPANSION_DIM + j]);
//             }
//             printf("\n");
//         }
        tiled_norm_auto(seq_len, expansion_dim, gelu_in, gelu_out,
                15.0 / (1 << 0), 0, expansion_dim, expansion_dim, IGELU);

        printf("GELU done.\n");
//         for(int i = 0; i < SEQ_LEN; i++) {
//             for(int j = 0; j < EXPANSION_DIM; j++) {
//                 printf("%d ", gelu_out[i * EXPANSION_DIM + j]);
//             }
//             printf("\n");
//         }
//         // FFN Layer 2: out = out_buf * ff2_w + ff2_b
//         tiled_matmul_auto(/*DIM_I=*/seq_len, /*DIM_J=*/hidden_dim, /*DIM_K=*/expansion_dim,
//             /*A=*/ ff2_in, /*B=*/ ff2_w,
//             /*D=*/ ff2_b, /*C=*/ ff2_out,
//             /*stride_A=*/expansion_dim, /*stride_B=*/hidden_dim, /*stride_D=*/hidden_dim, /*stride_C=*/hidden_dim);

//         printf("MatMul 2 done.\n");
//         for(int i = 0; i < SEQ_LEN; i++) {
//             for(int j = 0; j < HIDDEN_DIM; j++) {
//                 printf("%d ", ff2_out[i * HIDDEN_DIM + j]);
//             }
//             printf("\n");
//         }
//         // out = LN(out)
//         tiled_norm_auto(seq_len, hidden_dim, ln_in, ln_out,
//                 1.0 / (1 << 0), 0, hidden_dim, hidden_dim, LAYERNORM);

//         // printf("LayerNorm done.\n");
//         // for(int i = 0; i < SEQ_LEN; i++) {
//         //     for(int j = 0; j < HIDDEN_DIM; j++) {
//         //         printf("%d ", ln_out[i * HIDDEN_DIM + j]);
//         //     }
//         //     printf("\n");
//         // }

//         // out = out + input
//         tiled_add_auto(seq_len, hidden_dim,
//             ln_out, resadd_input, out, hidden_dim, hidden_dim, hidden_dim);
//         printf("ResAdd done.\n");
//         // for(int i = 0; i < SEQ_LEN; i++) {
//         //     for(int j = 0; j < HIDDEN_DIM; j++) {
//         //         printf("%d ", out[i * HIDDEN_DIM + j]);
//         //     }
//         //     printf("\n");
//         // }
}

/* ---------------------------------------------------  */
// ffn test end
/* ---------------------------------------------------  */

/* ---------------------------------------------------  */
// gelu test begin
/* ---------------------------------------------------  */


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
int32_t igelu_golden_q32_to_q16(acc_t x_q32, acc_scale_t scale, acc_scale_t bert_scale) {
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

int32_t gelu_golden_q32_to_int(acc_t x_q32, acc_scale_t scale, acc_scale_t bert_scale)
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
    int32_t C_in[SEQ_LEN][EXPANSION_DIM],      // 输入：Q32.0
    int32_t C_out[SEQ_LEN][EXPANSION_DIM],     // 输出：Q16.16
    float scale,
    float bert_scale
){
    for(int i=0;i<SEQ_LEN;i++){
        for(int j=0;j<EXPANSION_DIM;j++){
            // C_out[i][j] = scale_and_sat(C_in[i][j], IGELU, 1.0f, bert_scale);
            // C_out[i][j] = igelu_golden_q32_to_q16(C_in[i][j], scale, bert_scale);
            C_out[i][j] = gelu_golden_q32_to_int(C_in[i][j], scale, bert_scale);
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
    int64_t count = (int64_t)SEQ_LEN * (int64_t)EXPANSION_DIM;

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EXPANSION_DIM; j++) {
            int32_t g = ffn_gelu_out_g[i][j];
            int32_t t = ffn_gelu_out[i][j];
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

void change_ffn_gelu_out (acc_scale_t bert_scale) {
    double scale = (-0.2888 * bert_scale) / 4; // 1 / (2q1 * bert_scale)
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EXPANSION_DIM; j++) {
            ffn_gelu_out[i][j] = (int32_t)(ffn_gelu_out[i][j] * scale);
        }
    }   
}

void change_ffn_gelu_out_g (acc_scale_t bert_scale) {
    double scale = (-0.2888 * bert_scale) / 4; // 1 / (2q1 * bert_scale)
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EXPANSION_DIM; j++) {
            ffn_gelu_out_g[i][j] = (int32_t)(ffn_gelu_out_g[i][j] * scale);
        }
    }   
}

void dump_gelu_results(
    const int32_t ffn_gelu_in   [SEQ_LEN][EXPANSION_DIM],  // Q32.0
    const int32_t ffn_gelu_out  [SEQ_LEN][EXPANSION_DIM],  // Q16.16
    const int32_t ffn_gelu_out_g[SEQ_LEN][EXPANSION_DIM]   // Q16.16
) {
    const char *path =
        "/cluster/home/geyh/RV/xuantie_gnu_toolchain/test/rvme/tests/final_output/gelu_cp.out";

    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("fopen failed");
        return;
    }

    double mse = 0.0;
    double dot = 0.0;
    double norm_out = 0.0;
    double norm_golden = 0.0;

    /* === KL divergence === */
    double kl_div = 0.0;
    double sum_out = 0.0;
    double sum_golden = 0.0;
    const double eps = 1e-12;

    const long long total = (long long)SEQ_LEN * EXPANSION_DIM;

    /* ================= 输入矩阵 ================= */
    fprintf(fp, "===== GELU INPUT (Q32.0) =====\n");
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EXPANSION_DIM; j++) {
            fprintf(fp, "%d ", ffn_gelu_in[i][j]);
        }
        fprintf(fp, "\n");
    }

    /* ================= 输出矩阵 ================= */
    fprintf(fp, "\n===== GELU OUTPUT (Q16.16) =====\n");
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EXPANSION_DIM; j++) {
            fprintf(fp, "%d ", ffn_gelu_out[i][j]);
        }
        fprintf(fp, "\n");
    }

    /* ================= Golden 输出 ================= */
    fprintf(fp, "\n===== GELU GOLDEN OUTPUT (Q16.16) =====\n");
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EXPANSION_DIM; j++) {
            fprintf(fp, "%d ", ffn_gelu_out_g[i][j]);
        }
        fprintf(fp, "\n");
    }

    /* ================= 统计量第一遍 ================= */
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EXPANSION_DIM; j++) {
            double y  = (double)ffn_gelu_out[i][j];
            double yg = (double)ffn_gelu_out_g[i][j];

            double diff = y - yg;
            mse += diff * diff;

            dot += y * yg;
            norm_out += y * y;
            norm_golden += yg * yg;

            sum_out += fabs(y);
            sum_golden += fabs(yg);
        }
    }

    mse /= (double)total;

    double cos_sim =
        dot / (sqrt(norm_out) * sqrt(norm_golden) + eps);

    /* ================= KL divergence ================= */
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EXPANSION_DIM; j++) {
            double p = fabs((double)ffn_gelu_out[i][j]) / (sum_out + eps);
            double q = fabs((double)ffn_gelu_out_g[i][j]) / (sum_golden + eps);

            p += eps;
            q += eps;

            kl_div += p * log(p / q);
        }
    }

    /* ================= 汇总 ================= */
    fprintf(fp, "\n===== METRICS =====\n");
    fprintf(fp, "Average MSE       : %.10e\n", mse);
    fprintf(fp, "Average CosineSim : %.10f\n", cos_sim);
    fprintf(fp, "KL Divergence     : %.10e\n", kl_div);

    fclose(fp);

    printf("[DUMP] GELU results dumped to %s\n", path);
}


/* ---------------------------------------------------  */
// gelu test end
/* ---------------------------------------------------  */


int main() {
    printf("start test!\n");
    printf("Attention test start...\n");
    printf("SEQ_LEN=%d, HIDDEN_DIM=%d, NUM_HEADS=%d\n", SEQ_LEN, HIDDEN_DIM, NUM_HEADS);
    
    rand_init_matmul();
    // reorder_A();
    // reorder_Q_in();
    // pad_Q_inR_to_256();
    // load_C_from_file("/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/0120/number_quantized.txt");
    
    attention(HIDDEN_DIM, NUM_HEADS, SEQ_LEN,
        (const elem_t *)inputR, (const elem_t *)inputR,
        (const elem_t *)Wq, (const elem_t *)Wk, (const elem_t *)Wv,
        (const acc_t *)Wq_b, (const acc_t *)Wk_b, (const acc_t *)Wv_b,
        (acc_t *)Q_buf, (acc_t *)K_buf, (acc_t *)V_buf,
        (const elem_t *)Q_inR_pad, (const elem_t *)KT_in,
        (acc_t *)attn_buf, 
        (const acc_t *)softmax_in, (acc_t *)softmax_attn_buf,
        (const elem_t *)attn_in, (const elem_t *)V_in,
        (acc_t *)out_buf,
        (const elem_t *)out_in, (const elem_t *)Wo, (const acc_t *)Wo_b,
        (acc_t *)out_buf_acc,
        (const acc_t *)resadd_input0,
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

    // dump_qkv_report(
    //     "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/0120/QKV_cpo.out",

    //     (const elem_t *)input,
    //     (const elem_t *)Wq,
    //     (const elem_t *)Wk,
    //     (const elem_t *)Wv,
    //     (const acc_t  *)Wq_b,
    //     (const acc_t  *)Wk_b,
    //     (const acc_t  *)Wv_b,

    //     (const acc_t *)Q_buf,
    //     (const acc_t *)K_buf,
    //     (const acc_t *)V_buf,

    //     (const acc_t *)Q_buf_g,
    //     (const acc_t *)K_buf_g,
    //     (const acc_t *)V_buf_g
    // );

    // /* ---------------------------------------------------------- */

    // /* ---------------------------------------------------------- */
    // // attn_matmul_test
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
    // /* ---------------------------------------------------------- */

    // /* ---------------------------------------------------------- */
    // softmax_golden_all_heads(
    //     softmax_in,
    //     softmax_attn_buf_g,
    //     NUM_HEADS,
    //     SEQ_LEN,
    //     /* input_q_bits  */ 8,
    //     /* output_q_bits */ 16
    // );

    // dump_softmax_report(
    //     "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/0120/softmax_cp.out",
    //     softmax_in,
    //     softmax_attn_buf,
    //     softmax_attn_buf_g,
    //     NUM_HEADS,
    //     SEQ_LEN,
    //     16
    // );
    // /* ---------------------------------------------------------- */


    // /* ---------------------------------------------------------- */

    // // golden
    // resadd_golden_int32(
    //     resadd_input0,
    //     resadd_input,
    //     resadd_out_g,
    //     SEQ_LEN,
    //     HIDDEN_DIM
    // );

    // // dump + compare
    // dump_resadd_report(
    //     "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/0120/resadd_cp.out",
    //     resadd_input0,
    //     resadd_input,
    //     resadd_out,
    //     resadd_out_g
    // );

    // /* ---------------------------------------------------------- */

    // /* ---------------------------------------------------------- */
    // layernorm_golden_int32((const int32_t*)LN_in, (int32_t*)out_g, SEQ_LEN, HIDDEN_DIM, 1, 0);

    dump_layernorm_report_full(
        "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/0120/layernorm_cp.out",
        (const acc_t *)LN_in,           // 输入矩阵
        (const acc_t *)out,             // HW 输出  
        (const acc_t *)out_g,           // golden 输出
        SEQ_LEN,
        HIDDEN_DIM
    );
    // /* ---------------------------------------------------------- */

    // ffn(HIDDEN_DIM, EXPANSION_DIM, SEQ_LEN,
    //     (const elem_t *)ffn_input, (acc_t *)ffn_out,
    //     (const elem_t *)ffn_ff1_w, (const elem_t *)ffn_ff2_w,
    //     (const acc_t *)ffn_ff1_b, (const acc_t *)ffn_ff2_b,
    //     (const acc_t *)ffn_resadd_input,
    //     (acc_t *)ffn_ff2_in,
    //     (acc_t *)ffn_ff1_out, (acc_t *)ffn_ff2_out,
    //     (acc_t *)ffn_ln_in,
    //     (const acc_t *)ffn_gelu_in,
    //     (acc_t *)ffn_gelu_out, (acc_t *)ffn_ln_out);

    // printf("\nffn_gelu_out:\n");
    // for (int i = 0; i < SEQ_LEN; i++) {
    //     for (int j = 0; j < HIDDEN_DIM; j++) {
    //         printf("%6d ", ffn_gelu_out[i][j]);
    //     }
    //     printf("\n");
    // }

    // // 调用硬件/软件实现版本
    // tiled_norm_auto(SEQ_LEN, EXPANSION_DIM, (const acc_t*)ffn_gelu_in, (acc_t*)ffn_gelu_out,
    //                 5.0 / (1 << 0), 0, EXPANSION_DIM, EXPANSION_DIM, IGELU);

    // 计算黄金参考
    // golden_IGELU_matrix(ffn_gelu_in, ffn_gelu_out_g, 1.0f ,15.0f);

    // change_ffn_gelu_out(15.0f);
    // // change_ffn_gelu_out_g(0.02f);

    // dump_gelu_results(
    //     ffn_gelu_in,
    //     ffn_gelu_out,
    //     ffn_gelu_out_g
    // );

    return 0;
}



