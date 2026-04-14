cmake_minimum_required(VERSION 3.18)
project(cuda_operators LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/ops)

# Source files
file(GLOB CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/binding.cpp
)

# Create pybind module
pybind11_add_module(cuda_ops ${CUDA_SOURCES})

# Set properties for CUDA
set_target_properties(cuda_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link CUDA libraries
target_link_libraries(cuda_ops PRIVATE
    CUDA::cudart
    CUDA::cublas
)

# Print configuration
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
