#include "../common/norm_v2.h"
#include "../common/matadd.h"
#include "../common/matmul.h"

 
#include <stdio.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#define CACHELINE_SIZE 64
// #define SEQ_LEN 128
// #define HIDDEN_DIM 64
// #define EXPANSION_DIM (HIDDEN_DIM )


// #define SEQ_LEN 256
// #define HIDDEN_DIM 768
// #define EXPANSION_DIM 512
#define SEQ_LEN 256
#define HIDDEN_DIM 64
#define EXPANSION_DIM 64

static alignas(CACHELINE_SIZE) elem_t input[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t ff1_w[HIDDEN_DIM][EXPANSION_DIM] = {[0 ... HIDDEN_DIM-1][0 ... EXPANSION_DIM-1] = 1};
static alignas(CACHELINE_SIZE) elem_t ff2_w[EXPANSION_DIM][HIDDEN_DIM] = {[0 ... EXPANSION_DIM-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ff1_b[SEQ_LEN][EXPANSION_DIM] = {[0 ... SEQ_LEN-1][0 ... EXPANSION_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ff2_b[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ff1_out[SEQ_LEN][EXPANSION_DIM] = {[0 ... SEQ_LEN-1][0 ... EXPANSION_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ff2_out[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t gelu_out[SEQ_LEN][EXPANSION_DIM] = {[0 ... SEQ_LEN-1][0 ... EXPANSION_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ln_out[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t out[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t resadd_input[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ff2_in[SEQ_LEN][EXPANSION_DIM] = {[0 ... SEQ_LEN-1][0 ... EXPANSION_DIM-1] = 1};
static alignas(CACHELINE_SIZE) acc_t ln_in[SEQ_LEN][HIDDEN_DIM] = {[0 ... SEQ_LEN-1][0 ... HIDDEN_DIM-1] = 17};

void ffn (int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, acc_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,
        const acc_t * resadd_input,
        acc_t * ff2_in,
        acc_t * ff1_out, acc_t * ff2_out,
        acc_t * ln_in,
        acc_t * gelu_out, acc_t * ln_out
){

        // // FFN Layer 1: out_buf = GELU(input * ff1_w + ff1_b)
        // tiled_matmul_auto(/*DIM_I=*/seq_len, /*DIM_J=*/expansion_dim, /*DIM_K=*/hidden_dim,
        //     /*A=*/ input, /*B=*/ ff1_w,
        //     /*D=*/ ff1_b, /*C=*/ ff1_out,
        //     /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim);
        
        // printf("MatMul 1 done.\n");
        
        // // for(int i = 0; i < SEQ_LEN; i++) {
        // //     for(int j = 0; j < EXPANSION_DIM; j++) {
        // //         printf("%d ", ff1_out[i * EXPANSION_DIM + j]);
        // //     }
        // //     printf("\n");
        // // }

        // tiled_norm_auto(seq_len, expansion_dim, ff1_out, gelu_out,
        //         0.02 / (1 << 0), 0, expansion_dim, expansion_dim, IGELU);

        // printf("GELU done.\n");
        
        // // for(int i = 0; i < SEQ_LEN; i++) {
        // //     for(int j = 0; j < EXPANSION_DIM; j++) {
        // //         printf("%d ", gelu_out[i * EXPANSION_DIM + j]);
        // //     }
        // //     printf("\n");
        // // }

        // // FFN Layer 2: out = out_buf * ff2_w + ff2_b
        // tiled_matmul_auto(/*DIM_I=*/seq_len, /*DIM_J=*/hidden_dim, /*DIM_K=*/expansion_dim,
        //     /*A=*/ ff2_in, /*B=*/ ff2_w,
        //     /*D=*/ ff2_b, /*C=*/ ff2_out,
        //     /*stride_A=*/expansion_dim, /*stride_B=*/hidden_dim, /*stride_D=*/hidden_dim, /*stride_C=*/hidden_dim);

        // printf("MatMul 2 done.\n");

        // // for(int i = 0; i < SEQ_LEN; i++) {
        // //     for(int j = 0; j < HIDDEN_DIM; j++) {
        // //         printf("%d ", ff2_out[i * HIDDEN_DIM + j]);
        // //     }
        // //     printf("\n");
        // // }
        
        // // out = LN(out)
        // tiled_norm_auto(seq_len, hidden_dim, ln_in, ln_out,
        //         1.0 / (1 << 0), 0, hidden_dim, hidden_dim, LAYERNORM);

        // printf("LayerNorm done.\n");

        // for(int i = 0; i < SEQ_LEN; i++) {
        //     for(int j = 0; j < HIDDEN_DIM; j++) {
        //         printf("%d ", ln_out[i * HIDDEN_DIM + j]);
        //     }
        //     printf("\n");
        // }
        
        // out = out + input
        tiled_add_auto(seq_len, hidden_dim,
            ln_out, resadd_input, out, hidden_dim, hidden_dim, hidden_dim);

        printf("ResAdd done.\n");

        // for(int i = 0; i < SEQ_LEN; i++) {
        //     for(int j = 0; j < HIDDEN_DIM; j++) {
        //         printf("%d ", out[i * HIDDEN_DIM + j]);
        //     }
        //     printf("\n");
        // }
}

// layernorm

void load_C_from_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening input file");
        exit(1);
    }

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            if (fscanf(fp, "%d", &ln_in[i][j]) != 1) {
                fprintf(stderr, "Error: not enough data for row %d col %d\n", i, j);
                fclose(fp);
                exit(1);
            }
        }
    }

    fclose(fp);
    printf("Successfully loaded C_S from %s\n", filename);
}


int main() {
    printf("start test!\n");
    printf("FFN test start...\n");
    printf("SEQ_LEN=%d, HIDDEN_DIM=%d, EXPANSION_DIM=%d\n", SEQ_LEN, HIDDEN_DIM, EXPANSION_DIM);

    // load_C_from_file("/cluster/home/guoym/gem5_matrix_engine/tests/cwq/0120/number_quantized.txt");  
	ffn(HIDDEN_DIM, EXPANSION_DIM, SEQ_LEN,
			(const elem_t *)input, (acc_t *)out,
			(const elem_t *)ff1_w, (const elem_t *)ff2_w,
			(const acc_t *)ff1_b, (const acc_t *)ff2_b,
            (const acc_t *)resadd_input,
			(acc_t *)ff2_in,
			(acc_t *)ff1_out, (acc_t *)ff2_out,
            (acc_t *)ln_in,
			(acc_t *)gelu_out, (acc_t *)ln_out);

	printf("FFN test end!\n");

    printf("\nresadd:\n");
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            printf("%6d ", out[i][j]);
        }
        printf("\n");
    }

	return 0;

}