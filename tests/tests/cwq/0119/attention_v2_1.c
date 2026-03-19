#include "/cluster/home/geyh/RV/xuantie_gnu_toolchain/test/rvme/include/matmul_v2_2.h"
#include "/cluster/home/geyh/RV/xuantie_gnu_toolchain/test/rvme/include/norm_v3_2.h"
#include "/cluster/home/geyh/RV/xuantie_gnu_toolchain/test/rvme/include/matadd_v2_2.h"
#include "/cluster/home/geyh/RV/xuantie_gnu_toolchain/test/rvme/include/fileio.h"
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
static alignas(CACHELINE_SIZE) acc_t out[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};

static alignas(CACHELINE_SIZE) acc_t Q_buf_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t K_buf_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t V_buf_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t attn_buf_cp[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {[0 ... NUM_HEADS-1][0 ... SEQ_LEN-1][0 ... SEQ_LEN-1] = 1};
static alignas(CACHELINE_SIZE) acc_t softmax_attn_buf_cp[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {[0 ... NUM_HEADS-1][0 ... SEQ_LEN-1][0 ... SEQ_LEN-1] = 1};
static alignas(CACHELINE_SIZE) acc_t out_buf_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t out_buf_acc_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t resadd_out_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t out_cp[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};









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
    // attn = softmax(attn)
    for (int head = 0; head < num_heads; head++) {
        const acc_t * in = softmax_in + head * seq_len * seq_len;
        acc_t * out = softmax_attn_buf + head * seq_len * seq_len;
        tiled_norm_auto(seq_len, seq_len, in, out,
            1.0 / (1<<8), 16, seq_len, seq_len, SOFTMAX);
    }
    printf("softmax finished\n");
    // out_buf = attn * V
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = attn_in + head * seq_len * seq_len;
        const elem_t * B = V_in + head * hidden_dim_per_head;
        acc_t * C = out_buf + head * hidden_dim_per_head;
        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len, A, B, NULL, C,
            seq_len, hidden_dim, 0, hidden_dim);
    }
    printf("attn * V finished\n");
    // out_buf_acc = out_buf * Wo + Wo_b
    tiled_matmul_auto(/*DIM_I=*/seq_len, /*DIM_J=*/hidden_dim, /*DIM_K=*/hidden_dim,
        /*A=*/ out_in, /*B=*/ Wo,
        /*D=*/ Wo_b, /*C=*/ out_buf_acc,
        /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/hidden_dim, /*stride_C=*/hidden_dim);
    printf("output matmul finished\n");

    // resadd_out = out_buf_acc + input
    tiled_add_auto(seq_len, hidden_dim,
        out_buf_acc, resadd_input, resadd_out, hidden_dim, hidden_dim, hidden_dim);
    // printf("resadd_out printed\n");

    // for (int i = 0; i < SEQ_LEN; i++) {
    //     for(int j = 0; j < HIDDEN_DIM; j++) {
    //         printf("%d ", resadd_out[i * HIDDEN_DIM + j]);
    //     }
    //     printf("\n");
    // }

    // out = LN(resadd_out)
    // tiled_norm_auto(seq_len, hidden_dim, resadd_out, out,
    //         1.0 / (1<<8), 16, seq_len, hidden_dim, LAYERNORM);
}

int main() {
    printf("start test!\n");
    printf("Attention test start...\n");
    printf("SEQ_LEN=%d, HIDDEN_DIM=%d, NUM_HEADS=%d\n", SEQ_LEN, HIDDEN_DIM, NUM_HEADS);
    printf("in1: out_buf_acc addr=%p\n", out_buf_acc);
    printf("in2: resadd_input addr=%p\n", resadd_input);
    // for (int i = 0; i < HIDDEN_DIM; i++) {
    //     for (int j = 0; j < SEQ_LEN; j++) {
    //         KT_in[i][j] = j;
    //     }
    // }

    /* ================= load attention inputs ================= */
    const char *base =
        "/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_inputs/";

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "input.txt"),
        input, sizeof(elem_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "Wq.txt"),
        Wq, sizeof(elem_t), HIDDEN_DIM * HIDDEN_DIM);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "Wk.txt"),
        Wk, sizeof(elem_t), HIDDEN_DIM * HIDDEN_DIM);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "Wv.txt"),
        Wv, sizeof(elem_t), HIDDEN_DIM * HIDDEN_DIM);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "Wq_b.txt"),
        Wq_b, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "Wk_b.txt"),
        Wk_b, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "Wv_b.txt"),
        Wv_b, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "Q_in.txt"),
        Q_in, sizeof(elem_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "KT_in.txt"),
        KT_in, sizeof(elem_t), HIDDEN_DIM * SEQ_LEN);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "attn_in.txt"),
        attn_in, sizeof(elem_t), NUM_HEADS * SEQ_LEN * SEQ_LEN);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "softmax_in.txt"),
        softmax_in, sizeof(acc_t), NUM_HEADS * SEQ_LEN * SEQ_LEN);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "V_in.txt"),
        V_in, sizeof(elem_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "out_in.txt"),
        out_in, sizeof(elem_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "Wo.txt"),
        Wo, sizeof(elem_t), HIDDEN_DIM * HIDDEN_DIM);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "Wo_b.txt"),
        Wo_b, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt(strcat(strcpy((char[512]){}, base), "resadd_input.txt"),
        resadd_input, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);


    
    attention(HIDDEN_DIM, NUM_HEADS, SEQ_LEN,
        (const elem_t *)input, (const elem_t *)input,
        (const elem_t *)Wq, (const elem_t *)Wk, (const elem_t *)Wv,
        (const acc_t *)Wq_b, (const acc_t *)Wk_b, (const acc_t *)Wv_b,
        (acc_t *)Q_buf, (acc_t *)K_buf, (acc_t *)V_buf,
        (const elem_t *)Q_in, (const elem_t *)KT_in,
        (acc_t *)attn_buf, 
        (const acc_t *)softmax_in, (acc_t *)softmax_attn_buf,
        (const elem_t *)attn_in, (const elem_t *)V_in,
        (acc_t *)out_buf,
        (const elem_t *)out_in, (const elem_t *)Wo, (const acc_t *)Wo_b,
        (acc_t *)out_buf_acc,
        (const acc_t *)resadd_input,
        (acc_t *)resadd_out, (acc_t *)out
    );
    printf("Attention test end.\n");
    printf("end test!\n");

    /* ================= dump attention outputs ================= */

    // write_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/Q_buf.txt",
    //     Q_buf, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    // write_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/K_buf.txt",
    //     K_buf, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    // write_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/V_buf.txt",
    //     V_buf, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    // write_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/attn_buf.txt",
    //     attn_buf, sizeof(acc_t),
    //     NUM_HEADS * SEQ_LEN * SEQ_LEN);

    // write_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/softmax_attn_buf.txt",
    //     softmax_attn_buf, sizeof(acc_t),
    //     NUM_HEADS * SEQ_LEN * SEQ_LEN);

    // write_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/out_buf.txt",
    //     out_buf, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    // write_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/out_buf_acc.txt",
    //     out_buf_acc, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    // write_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/resadd_out.txt",
    //     resadd_out, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    // write_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/out.txt",
    //     out, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    

    read_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/Q_buf.txt",
        Q_buf_cp, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/K_buf.txt",
        K_buf_cp, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/V_buf.txt",
        V_buf_cp, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/attn_buf.txt",
        attn_buf_cp, sizeof(acc_t),
        NUM_HEADS * SEQ_LEN * SEQ_LEN);

    read_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/softmax_attn_buf.txt",
        softmax_attn_buf_cp, sizeof(acc_t),
        NUM_HEADS * SEQ_LEN * SEQ_LEN);

    read_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/out_buf.txt",
        out_buf_cp, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/out_buf_acc.txt",
        out_buf_acc_cp, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/resadd_out.txt",
        resadd_out_cp, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    read_tensor_txt("/cluster/home/geyh/RV/xuantie_gnu_toolchain/attn_outputs/out.txt",
        out_cp, sizeof(acc_t), SEQ_LEN * HIDDEN_DIM);

    int mismatch_count = 0;
    for(int i = 0; i < SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            if (Q_buf[i][j] != Q_buf_cp[i][j]) {
                printf("Mismatch in Q_buf at (%d, %d): %d != %d\n", i, j, Q_buf[i][j], Q_buf_cp[i][j]);
                mismatch_count++;
                break;
            }
            if (K_buf[i][j] != K_buf_cp[i][j]) {
                printf("Mismatch in K_buf at (%d, %d): %d != %d\n", i, j, K_buf[i][j], K_buf_cp[i][j]);
                mismatch_count++;
                break;
            }
            if (V_buf[i][j] != V_buf_cp[i][j]) {
                printf("Mismatch in V_buf at (%d, %d): %d != %d\n", i, j, V_buf[i][j], V_buf_cp[i][j]);
                mismatch_count++;
                break;
            }
            if (out_buf[i][j] != out_buf_cp[i][j]) {
                printf("Mismatch in out_buf at (%d, %d): %d != %d\n", i, j, out_buf[i][j], out_buf_cp[i][j]);
                mismatch_count++;
            }
            if (out_buf_acc[i][j] != out_buf_acc_cp[i][j]) {
                printf("Mismatch in out_buf_acc at (%d, %d): %d != %d\n", i, j, out_buf_acc[i][j], out_buf_acc_cp[i][j]);
                mismatch_count++;
            }
            if (resadd_out[i][j] != resadd_out_cp[i][j]) {
                printf("Mismatch in resadd_out at (%d, %d): %d != %d\n", i, j, resadd_out[i][j], resadd_out_cp[i][j]);
                mismatch_count++;
            }
            if (out[i][j] != out_cp[i][j]) {
                printf("Mismatch in out at (%d, %d): %d != %d\n", i, j, out[i][j], out_cp[i][j]);
                mismatch_count++;
            }
        }
    }
    if (mismatch_count == 0)
    {
        printf("All outputs match the reference outputs!\n");
    } else {
        printf("Total mismatches found: %d\n", mismatch_count);
    }
    


    return 0;
}






