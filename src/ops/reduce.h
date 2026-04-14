#pragma once

#include <cuda_runtime.h>

// Reduce sum operation variants
namespace reduce {

// Naive implementation - each thread handles one element
void reduce_sum_naive(const float* input, float* output, int n,
                      cudaStream_t stream = 0);

// Optimized with shared memory
void reduce_sum_shared(const float* input, float* output, int n,
                       cudaStream_t stream = 0);

// Two-pass algorithm for large arrays
void reduce_sum_twopass(const float* input, float* output, int n,
                        cudaStream_t stream = 0);

// Warp shuffle optimized
void reduce_sum_warp(const float* input, float* output, int n,
                     cudaStream_t stream = 0);

// Best implementation - auto-select strategy
void reduce_sum(const float* input, float* output, int n,
                cudaStream_t stream = 0);

} // namespace reduce
