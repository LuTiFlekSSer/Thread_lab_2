#include "task2.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>

//TODO все три целые числа:  n/p ,корень size, p,
void TASK2_run(LAB2_matrix const mat1, LAB2_matrix const mat2, int const rank, int const comm_size,
               LAB2_matrix *result) {
    int const s_P = (int)round(sqrt(comm_size)), shift = mat1.n / s_P;
    int const i = rank / s_P, j = rank % s_P, k = (i + j) % s_P;

    LAB2_matrix a, b, res, res2, c;
    matrix_alloc(mat1.n, mat2.m, &c);
    matrix_get_block(mat1, i, k, shift, &a);
    matrix_get_block(mat2, k, j, shift, &b);

    for (int l = 0; l < s_P; ++l) {
        matrix_mult(a, b, &res);
        matrix_get_block(c, i, j, shift, &res2);
        for (int x = 0; x < shift; ++x) {
            for (int y = 0; y < shift; ++y) {
                *matrix_get(c, i * shift + x, j * shift + y) = *matrix_get(res2, x, y) + *matrix_get(res, x, y);
            }
        }

        matrix_free(&res);
        matrix_free(&res2);

        MPI_Request send_request1, recv_request1, send_request2, recv_request2;

        MPI_Isend(a.array, shift * shift, MPI_DOUBLE, i * s_P + (j + s_P - 1) % s_P, 0, MPI_COMM_WORLD, &send_request1);
        MPI_Irecv(a.array, shift * shift, MPI_DOUBLE, i * s_P + (j + 1) % s_P, 0, MPI_COMM_WORLD, &recv_request1);
        MPI_Wait(&send_request1, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_request1, MPI_STATUS_IGNORE);

        MPI_Isend(b.array, shift * shift, MPI_DOUBLE, (i + s_P - 1) % s_P * s_P + j, 0, MPI_COMM_WORLD, &send_request2);
        MPI_Irecv(b.array, shift * shift, MPI_DOUBLE, (i + 1) % s_P * s_P + j, 0, MPI_COMM_WORLD, &recv_request2);
        MPI_Wait(&send_request2, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_request2, MPI_STATUS_IGNORE);
    }

    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, c.array, mat1.n * mat2.m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(c.array, NULL, mat1.n * mat2.m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        *result = c;
    } else {
        matrix_free(&c);
    }

    matrix_free(&a);
    matrix_free(&b);
}
