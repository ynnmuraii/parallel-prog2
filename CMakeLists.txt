cmake_minimum_required(VERSION 3.20)
project(parallel-prog2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(matrix1 lab1/matrix1.cpp)
set_target_properties(matrix1 PROPERTIES FOLDER "lab1")
add_custom_target(run_matrix1
    COMMAND matrix1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(matrix2 lab2/matrix2.cpp)
set_target_properties(matrix2 PROPERTIES FOLDER "lab2")
if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix2 OpenMP::OpenMP_CXX)
endif()
add_custom_target(run_matrix2
    COMMAND matrix2
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)