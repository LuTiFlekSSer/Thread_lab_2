#include "task3.h"

#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct TASK3_matrix {
    int64_t n, m;
    double *array;
} TASK3_matrix;

void matrix_alloc(int64_t const x, int64_t const y, TASK3_matrix *matrix) {
    matrix->n = x;
    matrix->m = y;
    matrix->array = calloc(x * y, sizeof(double));
}

void matrix_free(TASK3_matrix const *matrix) {
    free(matrix->array);
}

double* matrix_get(TASK3_matrix const matrix, int64_t const x, int64_t const y) {
    return &matrix.array[x + matrix.n * y];
}

void matrix_print(TASK3_matrix const matrix, int64_t const start_x, int64_t const start_y, int64_t const end_x,
                  int64_t const end_y) {
    for (int64_t i = start_x; i < end_x; ++i) {
        for (int64_t j = start_y; j < end_y; ++j) {
            printf("%lf ", *matrix_get(matrix, i, j));
        }
        printf("\n");
    }
}

double f(double const x, double const y) {
    return 0.;
}


int64_t min(int64_t const a, int64_t const b) {
    return a < b ? a : b;
}

void TASK3_run(int64_t const n_points, double const eps, double const temperature, double const std, int const rank,
               int const comm_size) {
    double const h = 1. / ((double)n_points - 1);
    TASK3_matrix F, U;
    matrix_alloc(n_points, n_points, &F);
    matrix_alloc(n_points + 2, n_points + 2, &U);

    if (rank == 0) {
        for (int64_t i = 0; i < n_points; ++i) {
            for (int64_t j = 0; j < n_points; ++j) {
                if (i == 0) {
                    *matrix_get(U, 0, j + 1) = *matrix_get(U, n_points + 1, j + 1) = temperature;
                }

                *matrix_get(F, i, j) = f((double)i * h, (double)j * h);
                *matrix_get(U, i + 1, j + 1) = ((double)rand() / RAND_MAX * 2 - 1) * std;
            }
            *matrix_get(U, i, 0) = *matrix_get(U, i, n_points + 1) = temperature;
        }
        *matrix_get(U, n_points, 0) = *matrix_get(U, n_points, n_points + 1) = *matrix_get(U, n_points + 1, 0) =
            *matrix_get(U, n_points + 1, n_points + 1) = temperature;
    }

    MPI_Bcast(U.array, (n_points + 2) * (n_points + 2), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(F.array, n_points * n_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *dm = calloc(n_points, sizeof(double));
    double d_max = 1;
    double const h2 = h * h;

    double *sendbuf = calloc(n_points, sizeof(double)),
           *recvbuf = calloc(n_points, sizeof(double));
    int *recvcounts = calloc(comm_size, sizeof(int)),
        *displs = calloc(comm_size, sizeof(int));

    while (d_max > eps) {
        d_max = 0;
        double local_d_max = 0;
        for (int i = 0; i < n_points; ++i) {
            dm[i] = 0;
        }

        for (int64_t n = 1; n < n_points + 1; ++n) {
            int64_t const start = n / comm_size * rank + min(rank, n % comm_size),
                          end = start + n / comm_size + (rank < n % comm_size ? 1 : 0);
            int64_t counter = 0;

            for (int64_t i = start + 1; i < end + 1; ++i) {
                int64_t const j = n - i + 1;

                double const tmp = *matrix_get(U, i, j);
                sendbuf[counter] = 0.25 * (*matrix_get(U, i - 1, j) + *matrix_get(U, i + 1, j) + *
                    matrix_get(U, i, j - 1) + *matrix_get(U, i, j + 1) - h2 * *matrix_get(F, i - 1, j - 1));

                double const d = fabs(tmp - sendbuf[counter++]);

                if (dm[i - 1] < d) {
                    dm[i - 1] = d;
                }
            }

            for (int64_t i = 0; i < comm_size; ++i) {
                recvcounts[i] = n / comm_size + (i < n % comm_size ? 1 : 0);
            }
            for (int64_t i = 1; i < comm_size; ++i) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }

            MPI_Allgatherv(sendbuf, counter, MPI_DOUBLE, recvbuf, recvcounts, displs,MPI_DOUBLE, MPI_COMM_WORLD);

            for (int64_t i = 1; i < n + 1; ++i) {
                int64_t const j = n - i + 1;
                *matrix_get(U, i, j) = recvbuf[i - 1];
            }
        }

        for (int64_t n = n_points - 1; n > 0; --n) {
            int64_t const start = n / comm_size * rank + min(rank, n % comm_size),
                          end = start + n / comm_size + (rank < n % comm_size ? 1 : 0);
            int64_t counter = 0;

            for (int64_t i = n_points - n + 1 + start; i < n_points - n + 1 + end; ++i) {
                int64_t const j = 2 * n_points + 1 - i - n;

                double const tmp = *matrix_get(U, i, j);
                sendbuf[counter] = 0.25 * (*matrix_get(U, i - 1, j) + *matrix_get(U, i + 1, j) + *
                    matrix_get(U, i, j - 1) + *matrix_get(U, i, j + 1) - h2 * *matrix_get(F, i - 1, j - 1));

                double const d = fabs(tmp - sendbuf[counter++]);

                if (dm[i - 1] < d) {
                    dm[i - 1] = d;
                }
            }

            for (int64_t i = 0; i < comm_size; ++i) {
                recvcounts[i] = n / comm_size + (i < n % comm_size ? 1 : 0);
            }
            for (int64_t i = 1; i < comm_size; ++i) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }

            MPI_Allgatherv(sendbuf, counter, MPI_DOUBLE, recvbuf, recvcounts, displs,MPI_DOUBLE, MPI_COMM_WORLD);

            for (int64_t i = n_points - n + 1; i < n_points + 1; ++i) {
                int64_t const j = 2 * n_points + 1 - i - n;
                *matrix_get(U, i, j) = recvbuf[i - n_points + n - 1];
            }
        }

        for (int i = 0; i < n_points; ++i) {
            if (local_d_max < dm[i]) {
                local_d_max = dm[i];
            }
        }

        MPI_Allreduce(&local_d_max, &d_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        FILE *file = fopen("mat.txt", "w+");

        for (int64_t i = 0; i < n_points; ++i) {
            for (int64_t j = 0; j < n_points; ++j) {
                fprintf(file, "%e ", *matrix_get(U, i + 1, j + 1));
            }
            fprintf(file, "\n");
        }

        fclose(file);
    }

    free(dm);
    matrix_free(&U);
    matrix_free(&F);
    free(recvbuf);
    free(recvcounts);
    free(displs);
    free(sendbuf);
}
