#include <stdio.h>
#include <mpi.h>
#include "src/task3/task3.h"

int main() {
    int rank, comm_size;
    MPI_Init(NULL,NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    TASK3_run(500, 1e-4, 0.3, 1, rank, comm_size);

    MPI_Finalize();
    return 0;
}
