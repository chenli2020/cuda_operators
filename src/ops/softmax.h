#pragma once

#include <cuda_runtime.h>

namespace softmax {

// Softmax operation along last dimension
// Shape: [batch, seq_len, features] or [N, features]

// Naive implementation
void softmax_naive(const float* input, float* output, int rows, int cols,
                   cudaStream_t stream = 0);

// Online softmax (numerically stable)
void softmax_online(const float* input, float* output, int rows, int cols,
                    cudaStream_t stream = 0);

// Warp-level optimized softmax
void softmax_warp(const float* input, float* output, int rows, int cols,
                  cudaStream_t stream = 0);

// Best implementation
void softmax(const float* input, float* output, int rows, int cols,
             cudaStream_t stream = 0);

} // namespace softmax
