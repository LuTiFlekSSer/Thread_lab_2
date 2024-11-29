#include "utils.h"
#include <stdio.h>

void matrix_alloc(int64_t const x, int64_t const y, LAB2_matrix *matrix) {
    matrix->n = x;
    matrix->m = y;
    matrix->array = calloc(x * y, sizeof(double));
}

void matrix_free(LAB2_matrix const *matrix) {
    free(matrix->array);
}

double* matrix_get(LAB2_matrix const matrix, int64_t const x, int64_t const y) {
    return &matrix.array[x + matrix.n * y];
}

void matrix_print(LAB2_matrix const matrix, int64_t const start_x, int64_t const start_y, int64_t const end_x,
                  int64_t const end_y) {
    for (int64_t i = start_x; i < end_x; ++i) {
        for (int64_t j = start_y; j < end_y; ++j) {
            printf("%lf ", *matrix_get(matrix, i, j));
        }
        printf("\n");
    }
}