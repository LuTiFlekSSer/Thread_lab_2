#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include "task3.h"
#include "../utils/utils.h"

double f(double const x, double const y) {
    return 0.;
}


int min(int const a, int const b) {
    return a < b ? a : b;
}

void TASK3_run(int const n_points, double const eps, double const temperature, double const std, int const rank,
               int const comm_size) {
    double const h = 1. / ((double)n_points - 1);
    LAB2_matrix F, U;
    matrix_alloc(n_points, n_points, &F);
    matrix_alloc(n_points + 2, n_points + 2, &U);

    if (rank == 0) {
        for (int i = 0; i < n_points; ++i) {
            for (int j = 0; j < n_points; ++j) {
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

    double dm = 0;
    double d_max = 1;
    double const h2 = h * h;
    int const s_P = (int)round(sqrt(comm_size)), shift = F.n / s_P, shift_sqr = shift * shift;
    int const i_P = rank / s_P, j_P = rank % s_P;

    double *sendbuf = calloc(shift_sqr, sizeof(double)),
           *recvbuf = calloc(n_points * n_points, sizeof(double));
    int *recvcounts = calloc(comm_size, sizeof(int)),
        *displs = calloc(comm_size, sizeof(int));

    for (int i = 0; i < comm_size; ++i) {
        recvcounts[i] = shift_sqr;
    }

    for (int i = 1; i < comm_size; ++i) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    while (d_max > eps) {
        d_max = 0;
        dm = 0;

        int const i0 = i_P * shift + 1, j0 = j_P * shift + 1;
        int counter = 0;

        for (int n = 0; n < shift; ++n) {
            for (int i = i0; i <= i0 + n; ++i) {
                int const j = j0 + n - i + i0;

                double const tmp = *matrix_get(U, i, j);
                *matrix_get(U, i, j) = sendbuf[counter] = 0.25 * (*matrix_get(U, i - 1, j) +
                    *matrix_get(U, i + 1, j) + *matrix_get(U, i, j - 1) + *matrix_get(U, i, j + 1) -
                    h2 * *matrix_get(F, i - 1, j - 1));

                double const d = fabs(tmp - sendbuf[counter++]);

                if (dm < d) {
                    dm = d;
                }
            }
        }

        for (int n = shift - 2; n >= 0; --n) {
            for (int i = i0 + shift - 1; i >= i0 + shift - n - 1; --i) {
                int const j = j0 + 2 * shift - 2 - n - (i - i0);

                double const tmp = *matrix_get(U, i, j);
                *matrix_get(U, i, j) = sendbuf[counter] = 0.25 * (*matrix_get(U, i - 1, j) +
                    *matrix_get(U, i + 1, j) + *matrix_get(U, i, j - 1) + *matrix_get(U, i, j + 1) -
                    h2 * *matrix_get(F, i - 1, j - 1));

                double const d = fabs(tmp - sendbuf[counter++]);

                if (dm < d) {
                    dm = d;
                }
            }
        }

        MPI_Allgatherv(sendbuf, counter, MPI_DOUBLE, recvbuf, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        counter = 0;
        for (int l_p = 0; l_p < comm_size; ++l_p) {
            int const l_i0 = l_p / s_P * shift + 1,
                      l_j0 = l_p % s_P * shift + 1;

            for (int n = 0; n < shift; ++n) {
                for (int i = l_i0; i <= l_i0 + n; ++i) {
                    int const j = l_j0 + n - i + l_i0;
                    *matrix_get(U, i, j) = recvbuf[counter++];
                }
            }

            for (int n = shift - 2; n >= 0; --n) {
                for (int i = l_i0 + shift - 1; i >= l_i0 + shift - n - 1; --i) {
                    int const j = l_j0 + 2 * shift - 2 - n - (i - l_i0);
                    *matrix_get(U, i, j) = recvbuf[counter++];
                }
            }
        }

        MPI_Allreduce(&dm, &d_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        FILE *file = fopen("mat.txt", "w+");

        for (int i = 0; i < n_points; ++i) {
            for (int j = 0; j < n_points; ++j) {
                fprintf(file, "%e ", *matrix_get(U, i + 1, j + 1));
            }
            fprintf(file, "\n");
        }

        fclose(file);
    }

    matrix_free(&U);
    matrix_free(&F);
    free(recvbuf);
    free(recvcounts);
    free(displs);
    free(sendbuf);
}
