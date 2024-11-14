cmake_minimum_required(VERSION 3.22)
project(Thread_lab_2 C)

set(CMAKE_C_STANDARD 11)

find_package(MPI REQUIRED)

add_executable(Thread_lab_2 main.c
        src/task2/task2.c
        src/task2/task2.h)

target_link_libraries(Thread_lab_2 PUBLIC MPI::MPI_C)