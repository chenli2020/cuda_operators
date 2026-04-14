#pragma once

#include <cuda_runtime.h>

namespace layernorm {

// LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
// Shape: [batch, seq_len, hidden_dim] or [N, hidden_dim]

// Naive implementation
void layernorm_naive(const float* input, const float* weight,
                     const float* bias, float* output, int rows, int cols,
                     float eps = 1e-5f, cudaStream_t stream = 0);

// Optimized with warp-level primitives
void layernorm_warp(const float* input, const float* weight,
                    const float* bias, float* output, int rows, int cols,
                    float eps = 1e-5f, cudaStream_t stream = 0);

// Vectorized memory access
void layernorm_vectorized(const float* input, const float* weight,
                          const float* bias, float* output, int rows, int cols,
                          float eps = 1e-5f, cudaStream_t stream = 0);

// Best implementation
void layernorm(const float* input, const float* weight, const float* bias,
               float* output, int rows, int cols, float eps = 1e-5f,
               cudaStream_t stream = 0);

} // namespace layernorm
