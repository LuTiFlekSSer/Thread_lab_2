#include "task3.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>


double f(double const x, double const y) {
    return 0.;
}

void print_krasivo(double **arr, int64_t const n, int64_t const m) {
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < m; ++j) {
            if (arr[i][j] >= 0) {
                printf(" %4.6lf ", arr[i][j]);
            } else {
                printf("%4.6lf ", arr[i][j]);
            }
        }

        printf("\n");
    }
}

void TASK3_run(int64_t const n_points, double const eps, double const temperature, double const std) {
    double const h = 1. / ((double)n_points - 1);
    double **F = calloc(n_points, sizeof(double*)),
           **U = calloc(n_points + 2, sizeof(double*));

    for (int64_t i = 0; i < n_points; ++i) {
        F[i] = calloc(n_points, sizeof(double));
        U[i] = calloc(n_points + 2, sizeof(double));
    }
    U[n_points] = calloc(n_points + 2, sizeof(double));
    U[n_points + 1] = calloc(n_points + 2, sizeof(double));

    for (int64_t i = 0; i < n_points; ++i) {
        for (int64_t j = 0; j < n_points; ++j) {
            if (i == 0) {
                U[0][j + 1] = U[n_points + 1][j + 1] = temperature;
            }

            F[i][j] = f((double)i * h, (double)j * h);
            U[i + 1][j + 1] = (double)rand() / RAND_MAX * 2 - 1;
        }
        U[i][0] = U[i][n_points + 1] = temperature;
    }
    U[n_points][0] = U[n_points][n_points + 1] = U[n_points + 1][0] = U[n_points + 1][n_points + 1] = temperature;

    double d_max = 1;
    double const h2 = h * h;

    while (d_max > eps) {
        d_max = 0;

        for (int64_t i = 1; i < n_points + 1; ++i) {
            for (int64_t j = 1; j < n_points + 1; ++j) {
                double const tmp = U[i][j];
                U[i][j] = 0.25 * (U[i - 1][j] + U[i + 1][j] + U[i][j - 1] + U[i][j + 1] - h2 * F[i - 1][j - 1]);

                double const dm = fabs(tmp - U[i][j]);

                if (d_max < dm) {
                    d_max = dm;
                }
            }
        }
    }

    FILE *file = fopen("mat.txt", "w+");

    for (int64_t i = 0; i < n_points; ++i) {
        for (int64_t j = 0; j < n_points; ++j) {
            fprintf(file, "%e ", U[i + 1][j + 1]);
        }
        fprintf(file, "\n");
    }

    fclose(file);

    for (int64_t i = 0; i < n_points; ++i) {
        free(F[i]);
        free(U[i]);
    }
    free(U[n_points]);
    free(U[n_points + 1]);
    free(U);
    free(F);
}
