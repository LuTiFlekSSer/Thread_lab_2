cmake_minimum_required(VERSION 3.22)
project(Thread_lab_2 C)

set(CMAKE_C_STANDARD 11)

find_package(MPI)

if (NOT MPI_FOUND)
    include_directories(/usr/local/mpich/include)
    link_directories(/usr/local/mpich/lib)

    set(MPI_C_INCLUDE_DIR /usr/local/mpich/include)
    set(MPI_C_LIBRARIES /usr/local/mpich/lib/libmpi.so)
    set(MPI_C_COMPILER /usr/local/mpich/bin/mpicc)

    find_package(MPI REQUIRED)
endif()

add_executable(Thread_lab_2 main.c
        src/task1/task1.c
        src/task1/task1.h
        src/task2/task2.c
        src/task2/task2.h
        src/task3/task3.c
        src/task3/task3.h
        src/utils/utils.c
        src/utils/utils.h)

target_link_libraries(Thread_lab_2 PUBLIC MPI::MPI_C)
target_link_libraries(Thread_lab_2 PUBLIC m)