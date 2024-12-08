#include <mpi.h>
#include <stdlib.h>
#include "task1.h"

void TASK1_run(LAB2_matrix const mat, LAB2_matrix const vec, int const rank, int const comm_size, int const choice,
               LAB2_matrix *result) {
    switch (choice) {
    case 1: {
        int const step = mat.n / comm_size, start = rank * step,
                  end = rank == comm_size - 1 ? mat.n : (rank + 1) * step;

        int *recvc = calloc(comm_size, sizeof(int)),
            *displ = calloc(comm_size, sizeof(int));


        if (rank == comm_size - 1) {
            for (int i = 0; i < comm_size - 1; ++i) {
                recvc[i] = step;
            }
            recvc[comm_size - 1] = end - start;

            for (int i = 1; i < comm_size; ++i) {
                displ[i] = displ[i - 1] + recvc[i - 1];
            }
        }

        MPI_Bcast(recvc, comm_size, MPI_INT, comm_size - 1, MPI_COMM_WORLD);
        MPI_Bcast(displ, comm_size, MPI_INT, comm_size - 1, MPI_COMM_WORLD);


        double *tmp = calloc(end - start, sizeof(double));

        for (int i = start; i < end; i++) {
            for (int j = 0; j < mat.m; j++) {
                tmp[i - start] += *matrix_get(mat, i, j) * *matrix_get(vec, j, 0);
            }
        }


        LAB2_matrix ans;
        matrix_alloc(mat.n, 1, &ans);

        MPI_Gatherv(tmp, end - start, MPI_DOUBLE, ans.array, recvc, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);


        if (rank == 0) {
            *result = ans;
        } else {
            matrix_free(&ans);
        }
        free(tmp);
        free(recvc);
        free(displ);
        break;
    }
    case 2: {
        int const step = mat.m / comm_size, start = rank * step,
                  end = rank == comm_size - 1 ? mat.m : (rank + 1) * step;

        double *tmp = calloc(mat.n, sizeof(double)),
               *big_result = calloc(mat.n * comm_size, sizeof(double));

        for (int i = 0; i < mat.n; i++) {
            for (int j = start; j < end; j++) {
                tmp[i] += *matrix_get(mat, i, j) * *matrix_get(vec, j, 0);
            }
        }

        LAB2_matrix ans;
        matrix_alloc(mat.n, 1, &ans);

        MPI_Gather(tmp, mat.n, MPI_DOUBLE, big_result, mat.n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < mat.n; ++i) {
            for (int j = 0; j < comm_size; ++j) {
                *matrix_get(ans, i, 0) = *matrix_get(ans, i, 0) + big_result[j * mat.n + i];
            }
        }

        if (rank == 0) {
            *result = ans;
        } else {
            matrix_free(&ans);
        }

        free(tmp);
        free(big_result);
        break;
    }
    case 3: {
        break;
    }
    }
}
