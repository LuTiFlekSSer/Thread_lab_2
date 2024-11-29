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

void matrix_mult(LAB2_matrix const mat1, LAB2_matrix const mat2, LAB2_matrix *result) {
    matrix_alloc(mat1.n, mat2.m, result);

    for (int64_t i = 0; i < mat1.n; ++i) {
        for (int64_t j = 0; j < mat2.m; ++j) {
            double sum = 0;
            for (int64_t k = 0; k < mat1.m; ++k) {
                sum += *matrix_get(mat1, i, k) * *matrix_get(mat2, k, j);
            }
            *matrix_get(*result, i, j) = sum;
        }
    }
}

void matrix_get_block(LAB2_matrix const matrix, int64_t const i, int64_t const j, int64_t const block_size,
                      LAB2_matrix *block) {
    matrix_alloc(block_size, block_size, block);

    for (int64_t x = 0; x < block_size; ++x) {
        for (int64_t y = 0; y < block_size; ++y) {
            *matrix_get(*block, x, y) = *matrix_get(matrix, i * block_size + x, j * block_size + y);
        }
    }
}

void matrix_set_block(LAB2_matrix const matrix, LAB2_matrix const block, int64_t const i, int64_t const j) {
    for (int64_t x = 0; x < block.n; ++x) {
        for (int64_t y = 0; y < block.m; ++y) {
            *matrix_get(matrix, i * block.n + x, j * block.m + y) = *matrix_get(block, x, y);
        }
    }
}
