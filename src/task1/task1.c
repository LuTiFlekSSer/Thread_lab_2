#include <mpi.h>
#include <stdlib.h>
#include <math.h>
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

        double *tmp = calloc(mat.n, sizeof(double));

        for (int i = 0; i < mat.n; i++) {
            for (int j = start; j < end; j++) {
                tmp[i] += *matrix_get(mat, i, j) * *matrix_get(vec, j, 0);
            }
        }

        LAB2_matrix ans;
        matrix_alloc(mat.n, 1, &ans);

        MPI_Reduce(tmp, ans.array, mat.n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            *result = ans;
        } else {
            matrix_free(&ans);
        }

        free(tmp);
        break;
    }
    case 3: {
        int const step_p = (int)round(sqrt(comm_size));
        int const step_x = mat.n / step_p;
        int const step_y = mat.m / step_p;
        int const start_x = rank / step_p * step_x;
        int const start_y = rank % step_p * step_y;
        int const end_x = rank / step_p == step_p - 1 ? mat.n : (rank / step_p + 1) * step_x;
        int const end_y = rank % step_p == step_p - 1 ? mat.m : (rank % step_p + 1) * step_y;


        double *tmp = calloc(end_x - start_x, sizeof(double)),
               *recv = calloc(end_x - start_x, sizeof(double));
        int *recvc = calloc(step_p, sizeof(int)),
            *displ = calloc(step_p, sizeof(int));

        for (int r = 0; r < comm_size; r += step_p) {
            int const start_x_l = r / step_p * step_x;
            int const end_x_l = r / step_p == step_p - 1 ? mat.n : (r / step_p + 1) * step_x;
            recvc[r / step_p] = end_x_l - start_x_l;
        }


        for (int i = 1; i < step_p; ++i) {
            displ[i] = displ[i - 1] + recvc[i - 1];
        }

        for (int i = start_x; i < end_x; i++) {
            for (int j = start_y; j < end_y; j++) {
                tmp[i - start_x] += *matrix_get(mat, i, j) * *matrix_get(vec, j, 0);
            }
        }

        int group = rank / step_p;
        MPI_Comm subgroup_comm;
        MPI_Comm_split(MPI_COMM_WORLD, group, rank, &subgroup_comm);

        MPI_Reduce(tmp, recv, end_x - start_x,MPI_DOUBLE,MPI_SUM, 0, subgroup_comm);
        MPI_Comm_free(&subgroup_comm);

        LAB2_matrix ans;
        matrix_alloc(mat.n, 1, &ans);

        group = rank % step_p == 0 ? 1 : 0;
        MPI_Comm_split(MPI_COMM_WORLD, group, rank, &subgroup_comm);

        if (group) {
            MPI_Gatherv(recv, end_x - start_x,MPI_DOUBLE, ans.array, recvc, displ,MPI_DOUBLE, 0, subgroup_comm);
        }
        MPI_Comm_free(&subgroup_comm);

        if (rank == 0) {
            *result = ans;
        } else {
            matrix_free(&ans);
        }

        free(tmp);
        free(recv);
        free(recvc);
        free(displ);
        break;
    }
    }
}
