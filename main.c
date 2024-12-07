#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>
#include "src/task2/task2.h"
#include "src/task3/task3.h"
#include "src/utils/utils.h"


#define EPS 1e-6

void menu(int const rank, int const comm_size) {
    printf("-----------------------------\n");
    printf("| THREAD LAB2 WELCOMES YOU! |\n");
    printf("-----------------------------\n");

    char option;
    srand(time(0));

    while (1) {
        printf("\nChoose task:\n"
            "[1] - Task 1 (multiplying matrix by vector)\n"
            "[2] - Task 2 (multiplying matrix by matrix)\n"
            "[3] - Task 3 (solving the Dirichlet problem)\n"
            "[e] - Exit\n");

        scanf("%c", &option);
        getchar();

        switch (option) {
        case '1': {
            LAB2_matrix matrix, vector;

            while (1) {
                printf("Sizes of matrix (n, m) = ");
                scanf("%ld %ld", &matrix.n, &matrix.m);
                getchar();

                if (matrix.n <= 0 || matrix.m <= 0) {
                    printf("Sizes must be greater than 0\n");
                } else {
                    break;
                }
            }

            matrix_alloc(matrix.m, 1, &vector);
            matrix_alloc(matrix.n, matrix.m, &matrix);

            char custom;

            while (1) {
                printf("Do you want to enter operands? [y/n]\n");
                scanf("%c", &custom);
                getchar();

                if (custom == 'y') {
                    printf("Enter matrix\n");

                    for (int i = 0; i < matrix.n; ++i) {
                        for (int j = 0; j < matrix.m; ++j) {
                            scanf("%lf", matrix_get(matrix, i, j));
                        }
                    }
                    getchar();

                    printf("Enter vector\n");

                    for (int i = 0; i < matrix.m; ++i) {
                        scanf("%lf", matrix_get(vector, i, 0));
                    }
                    getchar();

                    break;
                } else if (custom == 'n') {
                    for (int i = 0; i < matrix.n; ++i) {
                        for (int j = 0; j < matrix.m; ++j) {
                            *matrix_get(matrix, i, j) = (double)rand() / RAND_MAX * 200 - 100;
                        }
                    }

                    for (int i = 0; i < matrix.m; ++i) {
                        *matrix_get(matrix, i, 0) = (double)rand() / RAND_MAX * 200 - 100;
                    }

                    break;
                } else {
                    printf("Invalid option\n");
                }
            }


            MPI_Bcast(&option, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
            MPI_Bcast(&matrix.n, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&matrix.m, 1, MPI_INT, 0, MPI_COMM_WORLD);

            MPI_Bcast(matrix.array, matrix.n * matrix.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(vector.array, vector.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            //TASK1_run #todo

            // if (custom == 'y') {
            //     matrix_print(result, 0, 0, result.n, result.m);
            // }

            matrix_free(&vector);
            matrix_free(&matrix);

            break;
        }
        case '2': {
            if (ceil(sqrt(comm_size)) - sqrt(comm_size) >= EPS) {
                printf("Number of processes must be perfect square\n");
                break;
            }
            LAB2_matrix matrix1, matrix2;

            while (1) {
                printf("Size of matrix (n) = ");
                scanf("%ld", &matrix1.n);
                getchar();

                if (matrix1.n <= 0) {
                    printf("Size must be greater than 0\n");
                } else if (matrix1.n % (int)round(sqrt(comm_size)) != 0) {
                    printf("Size of matrix must be divided by root of number of threads\n");
                } else {
                    break;
                }
            }

            matrix_alloc(matrix1.n, matrix1.n, &matrix2);
            matrix_alloc(matrix1.n, matrix1.n, &matrix1);

            char custom;

            while (1) {
                printf("Do you want to enter operands? [y/n]\n");

                scanf("%c", &custom);
                getchar();

                if (custom == 'y') {
                    printf("Enter first matrix\n");

                    for (int i = 0; i < matrix1.n; ++i) {
                        for (int j = 0; j < matrix1.m; ++j) {
                            scanf("%lf", matrix_get(matrix1, i, j));
                        }
                    }
                    getchar();

                    printf("Enter second matrix\n");

                    for (int i = 0; i < matrix1.n; ++i) {
                        for (int j = 0; j < matrix1.m; ++j) {
                            scanf("%lf", matrix_get(matrix2, i, j));
                        }
                    }
                    getchar();

                    break;
                } else if (custom == 'n') {
                    for (int i = 0; i < matrix1.n; ++i) {
                        for (int j = 0; j < matrix1.m; ++j) {
                            *matrix_get(matrix1, i, j) = (double)rand() / RAND_MAX * 200 - 100;
                        }
                    }

                    for (int i = 0; i < matrix1.n; ++i) {
                        for (int j = 0; j < matrix1.m; ++j) {
                            *matrix_get(matrix2, i, j) = (double)rand() / RAND_MAX * 200 - 100;
                        }
                    }

                    break;
                } else {
                    printf("Invalid option\n");
                }
            }

            MPI_Bcast(&option, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
            MPI_Bcast(&matrix1.n, 1, MPI_INT, 0, MPI_COMM_WORLD);

            MPI_Bcast(matrix1.array, matrix1.n * matrix1.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(matrix2.array, matrix2.n * matrix2.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            LAB2_matrix result;
            TASK2_run(matrix1, matrix2, rank, comm_size, &result);

            if (custom == 'y') {
                matrix_print(result, 0, 0, result.n, result.m);
            }

            matrix_free(&result);

            matrix_free(&matrix2);
            matrix_free(&matrix1);

            break;
        }
        case '3': {
            int n_points;
            double precision, border_temp, half_length;

            while (1) {
                printf("Number of grid points, precision, border temperature, half-length of distribution = ");

                scanf("%ld %lf %lf %lf", &n_points, &precision, &border_temp, &half_length);
                getchar();

                if (n_points <= 0 || precision <= 0 || half_length <= 0) {
                    printf("Number of grid points, precision, half-length of distribution must be greater than 0\n");
                } else {
                    break;
                }
            }
            MPI_Bcast(&option, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

            MPI_Bcast(&n_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&precision, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&border_temp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&half_length, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            TASK3_run(n_points, precision, border_temp, half_length, rank, comm_size);

            break;
        }
        case 'e': {
            MPI_Bcast(&option, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
            return;
        }
        default: {
            printf("Invalid option\n");
            break;
        }
        }
    }
}

void runner(int const rank, int const comm_size) {
    char command;
    while (1) {
        MPI_Bcast(&command, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

        switch (command) {
        case '1': {
            LAB2_matrix matrix, vector;
            MPI_Bcast(&matrix.n, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&matrix.m, 1, MPI_INT, 0, MPI_COMM_WORLD);

            matrix_alloc(matrix.m, 1, &vector);
            matrix_alloc(matrix.n, matrix.m, &matrix);

            MPI_Bcast(matrix.array, matrix.n * matrix.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(vector.array, vector.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            //TASK1_run #todo

            matrix_free(&matrix);
            matrix_free(&vector);

            break;
        }
        case '2': {
            LAB2_matrix matrix1, matrix2;
            MPI_Bcast(&matrix1.n, 1, MPI_INT, 0, MPI_COMM_WORLD);

            matrix_alloc(matrix1.n, matrix1.n, &matrix2);
            matrix_alloc(matrix1.n, matrix1.n, &matrix1);

            MPI_Bcast(matrix1.array, matrix1.n * matrix1.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(matrix2.array, matrix2.n * matrix2.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            TASK2_run(matrix1, matrix2, rank, comm_size, NULL);

            matrix_free(&matrix1);
            matrix_free(&matrix2);

            break;
        }
        case '3': {
            int n_points;
            double precision, border_temp, half_length;

            MPI_Bcast(&n_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&precision, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&border_temp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&half_length, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            TASK3_run(n_points, precision, border_temp, half_length, rank, comm_size);

            break;
        }
        case 'e': {
            return;
        }
        }
    }
}

int main() {
    int rank, comm_size;
    MPI_Init(NULL,NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        menu(rank, comm_size);
    } else {
        runner(rank, comm_size);
    }

    MPI_Finalize();
    // 500 0.0001 0.3 1
    return 0;
}
