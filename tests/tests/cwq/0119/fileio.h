#include <stdio.h>
#include <stdint.h>

/* === text tensor reader === */
static void read_tensor_txt(
    const char *path,
    void *buf,
    size_t elem_size,
    size_t elem_cnt
) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror(path);
        return;
    }

    if (elem_size == sizeof(elem_t)) {
        elem_t *p = (elem_t *)buf;
        for (size_t i = 0; i < elem_cnt; i++) {
            fscanf(fp, "%hhd", &p[i]);
        }
    } else if (elem_size == sizeof(acc_t)) {
        acc_t *p = (acc_t *)buf;
        for (size_t i = 0; i < elem_cnt; i++) {
            fscanf(fp, "%d", &p[i]);
        }
    }

    fclose(fp);
}

/* === text tensor writer === */
static void write_tensor_txt(
    const char *path,
    const void *buf,
    size_t elem_size,
    size_t elem_cnt
) {
    FILE *fp = fopen(path, "w");   // 不存在会自动创建
    if (!fp) {
        perror(path);
        return;
    }

    if (elem_size == sizeof(elem_t)) {
        const elem_t *p = (const elem_t *)buf;
        for (size_t i = 0; i < elem_cnt; i++) {
            fprintf(fp, "%d\n", p[i]);
        }
    } else if (elem_size == sizeof(acc_t)) {
        const acc_t *p = (const acc_t *)buf;
        for (size_t i = 0; i < elem_cnt; i++) {
            fprintf(fp, "%d\n", p[i]);
        }
    }

    fclose(fp);
}
