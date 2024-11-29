#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>

typedef struct LAB2_matrix {
    int64_t n, m;
    double *array;
} LAB2_matrix;

void matrix_alloc(int64_t x, int64_t y, LAB2_matrix *matrix);

void matrix_free(LAB2_matrix const *matrix);

double* matrix_get(LAB2_matrix matrix, int64_t x, int64_t y);

void matrix_print(LAB2_matrix matrix, int64_t start_x, int64_t start_y, int64_t end_x, int64_t end_y);

void matrix_mult(LAB2_matrix mat1, LAB2_matrix mat2, LAB2_matrix *result);

void matrix_get_block(LAB2_matrix matrix, int64_t i, int64_t j, int64_t block_size, LAB2_matrix *block);

void matrix_set_block(LAB2_matrix matrix, LAB2_matrix block, int64_t i, int64_t j);

#endif
