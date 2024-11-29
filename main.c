#include <stdio.h>
#include <mpi.h>

#include "src/task2/task2.h"
#include "src/task3/task3.h"
#include "src/utils/utils.h"


int main() {
    int rank, comm_size;
    MPI_Init(NULL,NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // TASK3_run(500, 1e-4, 0.3, 1, rank, comm_size);

    LAB2_matrix mat1, mat2, mat3;
    matrix_alloc(1002*5,1002*5, &mat1);
    matrix_alloc(1002*5, 1002*5, &mat2);

    for (int i = 0; i < mat1.n; ++i) {
        for (int j = 0; j < mat1.m; ++j) {
            *matrix_get(mat1, i, j) = i * 1002*5 + j;
            *matrix_get(mat2, i, j) = i * 1002*5 + j;
        }
    }

    TASK2_run(mat1, mat2, rank, comm_size, &mat3);
    if (rank == 0) {
        matrix_free(&mat3);
    }

    matrix_free(&mat1);
    matrix_free(&mat2);

    MPI_Finalize();
    return 0;
}
