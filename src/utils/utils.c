#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

void matrix_alloc(int const x, int const y, LAB2_matrix *matrix) {
    matrix->n = x;
    matrix->m = y;
    matrix->array = calloc(x * y, sizeof(double));
}

void matrix_free(LAB2_matrix const *matrix) {
    free(matrix->array);
}

double* matrix_get(LAB2_matrix const matrix, int const x, int const y) {
    return &matrix.array[x + matrix.n * y];
}

void matrix_print(LAB2_matrix const matrix, int const start_x, int const start_y, int const end_x, int const end_y) {
    for (int i = start_x; i < end_x; ++i) {
        for (int j = start_y; j < end_y; ++j) {
            printf("%lf ", *matrix_get(matrix, i, j));
        }
        printf("\n");
    }
}

void matrix_mult(LAB2_matrix const mat1, LAB2_matrix const mat2, LAB2_matrix *result) {
    matrix_alloc(mat1.n, mat2.m, result);

    for (int i = 0; i < mat1.n; ++i) {
        for (int j = 0; j < mat2.m; ++j) {
            double sum = 0;
            for (int k = 0; k < mat1.m; ++k) {
                sum += *matrix_get(mat1, i, k) * *matrix_get(mat2, k, j);
            }
            *matrix_get(*result, i, j) = sum;
        }
    }
}

void matrix_get_block(LAB2_matrix const matrix, int const i, int const j, int const block_size, LAB2_matrix *block) {
    matrix_alloc(block_size, block_size, block);

    for (int x = 0; x < block_size; ++x) {
        for (int y = 0; y < block_size; ++y) {
            *matrix_get(*block, x, y) = *matrix_get(matrix, i * block_size + x, j * block_size + y);
        }
    }
}
