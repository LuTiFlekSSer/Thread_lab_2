#include "task3.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <mpi_proto.h>
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

double f(double const x, double const y) {
    return 0.;
}

void custom_set_op_double(void *invec, void *inoutvec, int const *len, MPI_Datatype *datatype) {
    double const *in = (double*)invec;
    double *inout = (double*)inoutvec;

    int real_len;
    MPI_Type_size(*datatype, &real_len);
    real_len /= sizeof(double);

    for (int i = 0; i < real_len; i++) {
        if (!isnan(in[i])) {
            inout[i] = in[i];
        }
    }
}

void get_bibas(int const n_points, int const y, int *array_of_blocklengths, int *array_of_displacements) {
    for (int64_t i = 0; i < n_points; ++i) {
        array_of_blocklengths[i] = 1;
        array_of_displacements[i] = n_points + i * (y - 1) + y;
    }
}

void get_bobas(int const n_points, int const y, int *array_of_blocklengths, int *array_of_displacements) {
    int64_t j = 0;

    for (int64_t i = n_points - y + 2; i < y - 2; ++i) {
        array_of_blocklengths[j] = 1;
        array_of_displacements[j++] = n_points + i * (y - 1) + y;
    }
}

int64_t min(int64_t const a, int64_t const b) {
    return a < b ? a : b;
}

void TASK3_run(int64_t const n_points, double const eps, double const temperature, double const std, int const rank,
               int const comm_size) {
    double const h = 1. / ((double)n_points - 1);
    TASK3_matrix F, U, U_FAKE;
    matrix_alloc(n_points, n_points, &F);
    matrix_alloc(n_points + 2, n_points + 2, &U);
    matrix_alloc(n_points + 2, n_points + 2, &U_FAKE);

    if (rank == 0) {
        for (int64_t i = 0; i < n_points; ++i) {
            for (int64_t j = 0; j < n_points; ++j) {
                if (i == 0) {
                    *matrix_get(U, 0, j + 1) = *matrix_get(U, n_points + 1, j + 1) = temperature;
                    *matrix_get(U_FAKE, 0, j + 1) = *matrix_get(U_FAKE, n_points + 1, j + 1) = NAN;
                }

                *matrix_get(F, i, j) = f((double)i * h, (double)j * h);
                *matrix_get(U, i + 1, j + 1) = ((double)rand() / RAND_MAX * 2 - 1) * std;
                *matrix_get(U_FAKE, i + 1, j + 1) = NAN;
            }
            *matrix_get(U, i, 0) = *matrix_get(U, i, n_points + 1) = temperature;
            *matrix_get(U_FAKE, i, 0) = *matrix_get(U_FAKE, i, n_points + 1) = NAN;
        }
        *matrix_get(U, n_points, 0) = *matrix_get(U, n_points, n_points + 1) = *matrix_get(U, n_points + 1, 0) =
            *matrix_get(U, n_points + 1, n_points + 1) = temperature;
        *matrix_get(U_FAKE, n_points, 0) = *matrix_get(U_FAKE, n_points, n_points + 1) = *matrix_get(
                U_FAKE, n_points + 1, 0) =
            *matrix_get(U_FAKE, n_points + 1, n_points + 1) = NAN;
    }

    MPI_Bcast(U.array, (n_points + 2) * (n_points + 2), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(U_FAKE.array, (n_points + 2) * (n_points + 2), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(F.array, n_points * n_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *dm = calloc(n_points, sizeof(double));
    double d_max = 1;
    double const h2 = h * h;

    MPI_Op custom_op;
    MPI_Op_create(custom_set_op_double, 1, &custom_op);

    int *array_of_blocklengths = calloc(n_points, sizeof(int)),
        *array_of_displacements = calloc(n_points, sizeof(int));

    MPI_Datatype *types_inc = calloc(n_points, sizeof(MPI_Datatype)),
                 *types_dec = calloc(n_points, sizeof(MPI_Datatype));

    for (int i = 0; i < n_points; ++i) {
        get_bibas(i + 1, n_points + 2, array_of_blocklengths, array_of_displacements);
        MPI_Type_indexed(i + 1, array_of_blocklengths, array_of_displacements,MPI_DOUBLE, &types_inc[i]);
        MPI_Type_commit(&types_inc[i]);
    }

    for (int i = 0; i < n_points - 1; ++i) {
        get_bobas(n_points + i + 1, n_points + 2, array_of_blocklengths, array_of_displacements);
        MPI_Type_indexed(n_points - i - 1, array_of_blocklengths, array_of_displacements,MPI_DOUBLE, &types_dec[i]);
        MPI_Type_commit(&types_dec[i]);
    }

    while (d_max > eps) {
        d_max = 0;
        double local_d_max = 0;
        for (int i = 0; i < n_points; ++i) {
            dm[i] = 0;
        }

        for (int64_t n = 1; n < n_points + 1; ++n) {
            int64_t const start = n / comm_size * rank + min(rank, n % comm_size),
                          end = start + n / comm_size + (rank < n % comm_size ? 1 : 0);

            for (int64_t i = start + 1; i < end + 1; ++i) {
                int64_t const j = n - i + 1;

                // if (rank == 0 && n == 1 && i == 1 && j == 1) {
                //     printf("%lf\n", *matrix_get(U, 2, 1));
                //     exit(0);
                // }
                double const tmp = *matrix_get(U, i, j);
                *matrix_get(U_FAKE, i, j) = 0.25 * (*matrix_get(U, i - 1, j) + *matrix_get(U, i + 1, j) + *
                    matrix_get(U, i, j - 1) + *matrix_get(U, i, j + 1) - h2 * *matrix_get(F, i - 1, j - 1));

                double const d = fabs(tmp - *matrix_get(U_FAKE, i, j));

                if (dm[i - 1] < d) {
                    dm[i - 1] = d;
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            // if (n == 2) {
            //     printf("%lf %lf %lf\n", *matrix_get(U, 1, 1), *matrix_get(U, 1, 2), *matrix_get(U, 2, 1));
            // }
            if (n == 2) {
                printf("1 2 RANK=%d REAL %lf ISNAN=%d\n ", rank, *matrix_get(U, 1, 2),
                       isnan(*matrix_get(U, 1, 2)));
                printf("2 1 RANK=%d REAL %lf ISNAN=%d\n ", rank, *matrix_get(U, 2, 1),
                       isnan(*matrix_get(U, 2, 1)));
            }
            // MPI_Allreduce(U_FAKE.array, U.array, 1, types_inc[n - 1], custom_op, MPI_COMM_WORLD);
            MPI_Reduce(U_FAKE.array, U.array, 1, types_inc[n - 1], custom_op, 0, MPI_COMM_WORLD);
            MPI_Bcast(U.array, 1, types_inc[n - 1], 0, MPI_COMM_WORLD);

            if (n == 2) {
                printf("1 2 RANK=%d REAL %lf ISNAN=%d\n ", rank, *matrix_get(U, 1, 2),
                       isnan(*matrix_get(U, 1, 2)));
                printf("2 1 RANK=%d REAL %lf ISNAN=%d\n ", rank, *matrix_get(U, 2, 1),
                       isnan(*matrix_get(U, 2, 1)));
                exit(0);
            }
        }

        for (int64_t n = n_points - 1; n > 0; --n) {
            int64_t const start = n / comm_size * rank + min(rank, n % comm_size),
                          end = start + n / comm_size + (rank < n % comm_size ? 1 : 0);

            for (int64_t i = n_points - n + 1 + start; i < n_points - n + 1 + end; ++i) {
                int64_t const j = 2 * n_points + 1 - i - n;

                double const tmp = *matrix_get(U, i, j);
                *matrix_get(U_FAKE, i, j) = 0.25 * (*matrix_get(U, i - 1, j) + *matrix_get(U, i + 1, j) + *
                    matrix_get(U, i, j - 1) + *matrix_get(U, i, j + 1) - h2 * *matrix_get(F, i - 1, j - 1));

                double const d = fabs(tmp - *matrix_get(U_FAKE, i, j));

                if (dm[i - 1] < d) {
                    dm[i - 1] = d;
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            // MPI_Allreduce(U_FAKE.array, U.array, 1, types_dec[n_points - (n + 1)], custom_op, MPI_COMM_WORLD);
            MPI_Reduce(U_FAKE.array, U.array, 1, types_dec[n_points - (n + 1)], custom_op, 0, MPI_COMM_WORLD);
            MPI_Bcast(U.array, 1, types_dec[n_points - (n + 1)], 0, MPI_COMM_WORLD);
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

    for (int i = 0; i < n_points; ++i) {
        MPI_Type_free(&types_inc[i]);
    }
    for (int i = 0; i < n_points - 1; ++i) {
        MPI_Type_free(&types_dec[i]);
    }

    MPI_Op_free(&custom_op);
    free(array_of_blocklengths);
    free(array_of_displacements);
    free(dm);
    matrix_free(&U);
    matrix_free(&U_FAKE);
    matrix_free(&F);
    free(types_dec);
    free(types_inc);
}
