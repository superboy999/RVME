#include "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/common/matmul_v2_2_2.h"
#include "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/common/norm_v3_2.h"
#include "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/common/matadd_v2_2.h"
// #include "../include/layernorm_v2.h"
#include <stdio.h>
#include <stdalign.h>   
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

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
static alignas(CACHELINE_SIZE) acc_t out[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t Q_in_1[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t KT_in_1[HIDDEN_DIM][SEQ_LEN] = {[0 ... HIDDEN_DIM-1][0 ... SEQ_LEN-1] = 1};
static alignas(CACHELINE_SIZE) acc_t attn_buf[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {[0 ... NUM_HEADS-1][0 ... SEQ_LEN-1][0 ... SEQ_LEN-1] = 1};

void attention(int hidden_dim, int num_heads, int seq_len,
        const elem_t * input, const elem_t * enc_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv,
        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        acc_t * Q_buf, acc_t * K_buf, acc_t * V_buf,

        const elem_t * Q_in, const elem_t * KT_in,
        acc_t * attn_buf, acc_t * softmax_attn_buf,

        const elem_t * attn_in, const elem_t * V_in,
        acc_t * out_buf,

        const elem_t * out_in, const elem_t * Wo, const acc_t * Wo_b,
        acc_t * out_buf_acc,

        const acc_t * resadd_input,
        acc_t * resadd_out, acc_t * out
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
        const acc_t * qkv_bs[] = {Wq_b, Wk_b, Wk_b};
        acc_t * qkv_outs[] = {Q_buf, K_buf, V_buf};
        const elem_t * qkv_w = qkv_weights[i];
        const elem_t * qkv_in = qkv_ins[i];
        const acc_t * qkv_b = qkv_bs[i];
        acc_t * qkv_out = qkv_outs[i];
        tiled_matmul_auto(seq_len, hidden_dim, hidden_dim, qkv_in, 
            qkv_w, qkv_b, qkv_out, hidden_dim, hidden_dim, hidden_dim, hidden_dim);
    }

    // attn = Q * K^T
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = Q_in + head * hidden_dim_per_head;
        const elem_t * B = KT_in + head * hidden_dim_per_head * seq_len;
        acc_t * C = attn_buf + head * seq_len * seq_len;
        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head, A, B, NULL, C,
            hidden_dim, seq_len, 0, seq_len);
    }
    printf("QK^T finished\n");

    for (int i = 0; i < NUM_HEADS; i++) {
        // printf("i = %d \n", i);
        // int cnt = 0;
        for(int j = 0; j < SEQ_LEN; j++) {
            // cnt = 0;
            for(int k = 0; k < SEQ_LEN; k++) {
                if(attn_buf[i * SEQ_LEN * SEQ_LEN + j * SEQ_LEN + k] == 64 * k);                    
                else
                printf("%d ", attn_buf[i * SEQ_LEN * SEQ_LEN + j * SEQ_LEN + k]);
            }
            printf("\n");
        }
    }
    
    // attn = softmax(attn)
    for (int head = 0; head < num_heads; head++) {
        const acc_t * in = attn_buf + head * seq_len * seq_len;
        acc_t * out = softmax_attn_buf + head * seq_len * seq_len;
        tiled_norm_auto(seq_len, seq_len, in, out,
            1.0 / (1<<8), 16, seq_len, seq_len, SOFTMAX);
    }
    printf("softmax finished\n");

    for (int i = 0; i < NUM_HEADS; i++) {
        for(int j = 0; j < 2; j++) {
            // for(int k = 0; k < SEQ_LEN; k++) {
            //     // if(softmax_attn_buf[i * SEQ_LEN * SEQ_LEN + j * SEQ_LEN + k] != 14770518) {
            //         printf("%d ", softmax_attn_buf[i * SEQ_LEN * SEQ_LEN + j * SEQ_LEN + k]);
            //     // }
            // }
            // printf("\n");
        }
    }

    // out_buf = attn * V
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = attn_in + head * seq_len * seq_len;
        const elem_t * B = V_in + head * hidden_dim_per_head;
        acc_t * C = out_buf + head * hidden_dim_per_head;
        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len, A, B, NULL, C,
            seq_len, hidden_dim, 0, hidden_dim);
    }
    printf("attn * V finished\n");

    for (int i = 0; i < SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            if(out_buf[i * HIDDEN_DIM + j] != 128) {
                printf("%d ", out_buf[i * HIDDEN_DIM + j]);
            }
        }
        printf("\n");
    }

    // out_buf_acc = out_buf * Wo + Wo_b
    tiled_matmul_auto(/*DIM_I=*/seq_len, /*DIM_J=*/hidden_dim, /*DIM_K=*/hidden_dim,
        /*A=*/ out_in, /*B=*/ Wo,
        /*D=*/ Wo_b, /*C=*/ out_buf_acc,
        /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/hidden_dim, /*stride_C=*/hidden_dim);
    printf("output matmul finished\n");

    for (int i = 0; i < SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            if(out_buf_acc[i * HIDDEN_DIM + j] != 769 ) {
                printf("%d ", out_buf_acc[i * HIDDEN_DIM + j]);
            }
        }
        printf("\n");
    }
    
    // resadd_out = out_buf_acc + input
    tiled_add_auto(seq_len, hidden_dim,
        out_buf_acc, resadd_input, resadd_out, hidden_dim, hidden_dim, hidden_dim);
    printf("resadd_out printed\n");

    for (int i = 0; i < SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            if(resadd_out[i * HIDDEN_DIM + j] == 2);
            else
            printf("%d ", resadd_out[i * HIDDEN_DIM + j]);
        }
        printf("\n");
    }

    // out = LN(resadd_out)
    // tiled_norm_auto(seq_len, hidden_dim, resadd_out, out,
    //         1.0 / (1<<8), 16, seq_len, hidden_dim, LAYERNORM);
}

int main() {
    printf("start test!\n");
    printf("Attention test start...\n");
    printf("SEQ_LEN=%d, HIDDEN_DIM=%d, NUM_HEADS=%d\n", SEQ_LEN, HIDDEN_DIM, NUM_HEADS);
    int cnt = 0;
    // for(int i = 0; i < SEQ_LEN; i++) {
    //     for(int j = 0; j < HIDDEN_DIM; j++) {
    //         Q_in_1[i][j] = 2;
    //     }
    // }

    for (int i = 0; i < HIDDEN_DIM; i++) {
        for(int j = 0; j < SEQ_LEN; j++) {
            KT_in_1[i][j] = j;
        }
    }
        
    attention(HIDDEN_DIM, NUM_HEADS, SEQ_LEN,
        (const elem_t *)input, (const elem_t *)input,
        (const elem_t *)Wq, (const elem_t *)Wk, (const elem_t *)Wv,
        (const acc_t *)Wq_b, (const acc_t *)Wk_b, (const acc_t *)Wv_b,
        (acc_t *)Q_buf, (acc_t *)K_buf, (acc_t *)V_buf,
        (const elem_t *)Q_in_1, (const elem_t *)KT_in_1,
        (acc_t *)attn_buf, (acc_t *)softmax_attn_buf,
        (const elem_t *)attn_in, (const elem_t *)V_in,
        (acc_t *)out_buf,
        (const elem_t *)out_in, (const elem_t *)Wo, (const acc_t *)Wo_b,
        (acc_t *)out_buf_acc,
        (const acc_t *)resadd_input,
        (acc_t *)resadd_out, (acc_t *)out
    );
    printf("Attention test end.\n");
    printf("end test!\n");

    return 0;
}






