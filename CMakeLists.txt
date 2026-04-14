cmake_minimum_required(VERSION 3.18)
project(cuda_operators LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Find pybind11 - try multiple methods
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # Try to find pybind11 via pip
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.__file__)"
        OUTPUT_VARIABLE PYBIND11_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PYBIND11_PATH)
        get_filename_component(PYBIND11_DIR "${PYBIND11_PATH}" DIRECTORY)
        # Try to find and load pybind11 CMake
        find_file(PYBIND11_CMAKE "pybind11Tools.cmake" PATHS "${PYBIND11_DIR}/share/cmake/pybind11" "${PYBIND11_DIR}/cmake")
        if(PYBIND11_CMAKE)
            get_filename_component(PYBIND11_CMAKE_DIR "${PYBIND11_CMAKE}" DIRECTORY)
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
            find_package(pybind11 REQUIRED)
            message(STATUS "Found pybind11 CMake at: ${PYBIND11_CMAKE_DIR}")
        else()
            # Fallback: just include the headers
            include_directories(${PYBIND11_DIR}/include)
            message(STATUS "Found pybind11 headers at: ${PYBIND11_DIR}/include")
            set(PYBIND11_FOUND_BY_HEADER TRUE)
        endif()
    else()
        message(FATAL_ERROR "pybind11 not found. Please install: pip install pybind11")
    endif()
endif()

# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Find CUDA package for libraries
find_package(CUDA QUIET)

# Setup CUDA library paths
if(CUDA_FOUND)
    message(STATUS "Found CUDA: ${CUDA_VERSION}")
    message(STATUS "CUDA toolkit root: ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    # Try to find CUDA manually
    if(CMAKE_CUDA_COMPILER)
        get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
        message(STATUS "Deduced CUDA toolkit root: ${CUDA_TOOLKIT_ROOT_DIR}")
    endif()
endif()

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
if(PYBIND11_FOUND_BY_HEADER)
    # Manual module creation when only headers are available
    Python_add_module(cuda_ops ${CUDA_SOURCES})
    target_include_directories(cuda_ops PRIVATE ${PYBIND11_DIR}/include)
else()
    # Use pybind11 helper when full CMake support is available
    pybind11_add_module(cuda_ops ${CUDA_SOURCES})
endif()

# Set properties for CUDA
set_target_properties(cuda_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link CUDA libraries - multiple fallback methods
if(TARGET CUDA::cudart)
    # Modern CMake CUDA targets
    target_link_libraries(cuda_ops PRIVATE CUDA::cudart)
    if(TARGET CUDA::cublas)
        target_link_libraries(cuda_ops PRIVATE CUDA::cublas)
    endif()
    message(STATUS "Using modern CMake CUDA targets")
elseif(CUDA_LIBRARIES)
    # Legacy CUDA package
    target_link_libraries(cuda_ops PRIVATE ${CUDA_LIBRARIES})
    message(STATUS "Using legacy CUDA libraries: ${CUDA_LIBRARIES}")
else()
    # Manual library finding
    find_library(CUDART_LIB cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    find_library(CUBLAS_LIB cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

    if(CUDART_LIB)
        target_link_libraries(cuda_ops PRIVATE ${CUDART_LIB})
        message(STATUS "Found cudart: ${CUDART_LIB}")
    else()
        message(WARNING "cudart library not found, linking may fail")
    endif()

    if(CUBLAS_LIB)
        target_link_libraries(cuda_ops PRIVATE ${CUBLAS_LIB})
        message(STATUS "Found cublas: ${CUBLAS_LIB}")
    else()
        message(WARNING "cublas library not found, performance may be reduced")
    endif()
endif()

# Add CUDA library directories for runtime
if(CUDA_TOOLKIT_ROOT_DIR)
    target_link_directories(cuda_ops PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
endif()

# Print configuration
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
