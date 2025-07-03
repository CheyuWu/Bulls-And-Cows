cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0104 NEW)
project(cuda_bull_and_cows LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)

find_package(CUDA REQUIRED)
include_directories(include ${CUDA_INCLUDE_DIRS})

add_executable(bull_and_cows
    src/main.cpp
    src/utils.cpp
    src/player.cpp
    src/cuda_utils.cu
)
