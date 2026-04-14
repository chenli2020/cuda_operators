#pragma once

#include <cuda_runtime.h>

namespace rmsnorm {

// RMSNorm: x / sqrt(mean(x^2) + eps) * weight
// Simpler than LayerNorm (no mean subtraction)
// Used in LLaMA, Mistral, etc.

// Naive implementation
void rmsnorm_naive(const float* input, const float* weight, float* output,
                   int rows, int cols, float eps = 1e-5f,
                   cudaStream_t stream = 0);

// Warp-level optimized
void rmsnorm_warp(const float* input, const float* weight, float* output,
                  int rows, int cols, float eps = 1e-5f,
                  cudaStream_t stream = 0);

// Vectorized implementation
void rmsnorm_vectorized(const float* input, const float* weight, float* output,
                        int rows, int cols, float eps = 1e-5f,
                        cudaStream_t stream = 0);

// Best implementation
void rmsnorm(const float* input, const float* weight, float* output, int rows,
             int cols, float eps = 1e-5f, cudaStream_t stream = 0);

} // namespace rmsnorm
