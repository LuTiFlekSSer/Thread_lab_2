#ifndef UTILS_H
#define UTILS_H

typedef struct LAB2_matrix {
    int n, m;
    double *array;
} LAB2_matrix;

void matrix_alloc(int x, int y, LAB2_matrix *matrix);

void matrix_free(LAB2_matrix const *matrix);

double* matrix_get(LAB2_matrix matrix, int x, int y);

void matrix_print(LAB2_matrix matrix, int start_x, int start_y, int end_x, int end_y);

void matrix_mult(LAB2_matrix mat1, LAB2_matrix mat2, LAB2_matrix *result);

void matrix_get_block(LAB2_matrix matrix, int i, int j, int block_size, LAB2_matrix *block);

void matrix_get_pad_block(LAB2_matrix matrix, int i, int j, int block_size, LAB2_matrix *block);

#endif
