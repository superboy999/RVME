#include "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/common/matmul_v2_2.h"
#include "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/common/norm_v3_2.h"
#include "/cluster/home/guoym/gem5_matrix_engine/o3_v0.6/tests/cwq/common/matadd_v2_2.h"

#include <stdio.h>
#include <stdlib.h>

#define CACHELINE_SIZE 64
#define SEQ_LEN 128
#define HIDDEN_DIM 768
#define NUM_HEADS 12

/* ================= aligned malloc ================= */

static inline void *aligned_malloc(size_t size, size_t align)
{
    void *raw;
    uintptr_t aligned;

    raw = malloc(size + align + sizeof(void*));
    if (!raw) return NULL;

    aligned = ((uintptr_t)raw + sizeof(void*) + align - 1) & ~(align - 1);
    ((void**)aligned)[-1] = raw;

    return (void*)aligned;
}

#define ALLOC_AND_INIT(ptr, type, n)                          \
    do {                                                      \
        ptr = (type*)aligned_malloc(sizeof(type) * (n), 64);     \
        if (!ptr) {                                          \
            printf("alloc failed: %s\n", #ptr);              \
            exit(1);                                         \
        }                                                     \
        for (size_t _i = 0; _i < (n); _i++)                   \
            ptr[_i] = 1;                                     \
    } while (0)

int main()
{
    printf("Attention microbenchmark start\n");
    printf("SEQ_LEN=%d HIDDEN_DIM=%d NUM_HEADS=%d\n",
           SEQ_LEN, HIDDEN_DIM, NUM_HEADS);

    const int hidden_dim_per_head = HIDDEN_DIM / NUM_HEADS;

    /* =========================================================
     * Stage 0: input / weights (长期存在)
     * ========================================================= */
    elem_t *input, *Wq, *Wk, *Wv, *Wo;
    acc_t  *Wq_b, *Wk_b, *Wv_b, *Wo_b;

    ALLOC_AND_INIT(input, elem_t, SEQ_LEN * HIDDEN_DIM);

    ALLOC_AND_INIT(Wq, elem_t, HIDDEN_DIM * HIDDEN_DIM);
    ALLOC_AND_INIT(Wk, elem_t, HIDDEN_DIM * HIDDEN_DIM);
    ALLOC_AND_INIT(Wv, elem_t, HIDDEN_DIM * HIDDEN_DIM);
    ALLOC_AND_INIT(Wo, elem_t, HIDDEN_DIM * HIDDEN_DIM);

    ALLOC_AND_INIT(Wq_b, acc_t, SEQ_LEN * HIDDEN_DIM);
    ALLOC_AND_INIT(Wk_b, acc_t, SEQ_LEN * HIDDEN_DIM);
    ALLOC_AND_INIT(Wv_b, acc_t, SEQ_LEN * HIDDEN_DIM);
    ALLOC_AND_INIT(Wo_b, acc_t, SEQ_LEN * HIDDEN_DIM);

    /* =========================================================
     * Stage 1: Q / K / V  (可选，这里只演示结构)
     * ========================================================= */
    acc_t *Q_buf, *K_buf, *V_buf;

    ALLOC_AND_INIT(Q_buf, acc_t, SEQ_LEN * HIDDEN_DIM);
    ALLOC_AND_INIT(K_buf, acc_t, SEQ_LEN * HIDDEN_DIM);
    ALLOC_AND_INIT(V_buf, acc_t, SEQ_LEN * HIDDEN_DIM);

    /* 这里如果你需要真实 QKV matmul，可解注 */
    /*
    tiled_matmul_auto(SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM,
        input, Wq, Wq_b, Q_buf,
        HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM);
    */

    /* =========================================================
     * Stage 2: QK^T
     * ========================================================= */
    elem_t *Q_in, *KT_in;
    acc_t  *attn_buf;

    ALLOC_AND_INIT(Q_in, elem_t, SEQ_LEN * HIDDEN_DIM);
    ALLOC_AND_INIT(KT_in, elem_t, HIDDEN_DIM * SEQ_LEN);
    ALLOC_AND_INIT(attn_buf, acc_t, NUM_HEADS * SEQ_LEN * SEQ_LEN);

    printf("state");
    for (int head = 0; head < NUM_HEADS; head++) {
        const elem_t *A = Q_in  + head * hidden_dim_per_head;
        const elem_t *B = KT_in + head * hidden_dim_per_head * SEQ_LEN;
        acc_t *C = attn_buf + head * SEQ_LEN * SEQ_LEN;

        tiled_matmul_auto(SEQ_LEN, SEQ_LEN, hidden_dim_per_head,
            A, B, NULL, C,
            HIDDEN_DIM, SEQ_LEN, 0, SEQ_LEN);
    }
    printf("QK^T finished\n");

    free(Q_in);
    free(KT_in);

    /* =========================================================
     * Stage 3: softmax
     * ========================================================= */
    acc_t *softmax_attn_buf;
    ALLOC_AND_INIT(softmax_attn_buf, acc_t, NUM_HEADS * SEQ_LEN * SEQ_LEN);

    for (int head = 0; head < NUM_HEADS; head++) {
        tiled_norm_auto(SEQ_LEN, SEQ_LEN,
            attn_buf + head * SEQ_LEN * SEQ_LEN,
            softmax_attn_buf + head * SEQ_LEN * SEQ_LEN,
            1.0 / (1<<8), 16,
            SEQ_LEN, SEQ_LEN, SOFTMAX);
    }
    printf("softmax finished\n");

    free(attn_buf);

    /* =========================================================
     * Stage 4: attn * V
     * ========================================================= */
    elem_t *attn_in, *V_in;
    acc_t  *out_buf;

    ALLOC_AND_INIT(attn_in, elem_t, NUM_HEADS * SEQ_LEN * SEQ_LEN);
    ALLOC_AND_INIT(V_in, elem_t, SEQ_LEN * HIDDEN_DIM);
    ALLOC_AND_INIT(out_buf, acc_t, SEQ_LEN * HIDDEN_DIM);

    for (int head = 0; head < NUM_HEADS; head++) {
        tiled_matmul_auto(SEQ_LEN, hidden_dim_per_head, SEQ_LEN,
            attn_in + head * SEQ_LEN * SEQ_LEN,
            V_in   + head * hidden_dim_per_head,
            NULL,
            out_buf + head * hidden_dim_per_head,
            SEQ_LEN, HIDDEN_DIM, 0, HIDDEN_DIM);
    }
    printf("attn * V finished\n");

    free(attn_in);
    free(V_in);
    free(softmax_attn_buf);

    /* =========================================================
     * Stage 5: output projection
     * ========================================================= */
    elem_t *out_in;
    acc_t  *out_buf_acc;

    ALLOC_AND_INIT(out_in, elem_t, SEQ_LEN * HIDDEN_DIM);
    ALLOC_AND_INIT(out_buf_acc, acc_t, SEQ_LEN * HIDDEN_DIM);

    tiled_matmul_auto(SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM,
        out_in, Wo, Wo_b, out_buf_acc,
        HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM);

    printf("output matmul finished\n");

    free(out_in);
    free(out_buf);
    free(Q_buf);
    free(K_buf);
    free(V_buf);

    /* =========================================================
     * Stage 6: residual add
     * ========================================================= */
    acc_t *resadd_out;
    ALLOC_AND_INIT(resadd_out, acc_t, SEQ_LEN * HIDDEN_DIM);

    tiled_add_auto(SEQ_LEN, HIDDEN_DIM,
        out_buf_acc, (acc_t*)input, resadd_out,
        HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM);

    printf("residual add finished\n");

    free(out_buf_acc);
    free(resadd_out);

    /* =========================================================
     * cleanup persistent
     * ========================================================= */
    free(input);
    free(Wq); free(Wk); free(Wv); free(Wo);
    free(Wq_b); free(Wk_b); free(Wv_b); free(Wo_b);

    printf("Attention microbenchmark end\n");
    return 0;
}
