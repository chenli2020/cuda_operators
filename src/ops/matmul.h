#pragma once

#include <cuda_runtime.h>

namespace matmul {

// Matrix multiplication: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]

// Naive implementation
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K,
                  cudaStream_t stream = 0);

// Shared memory with blocking
void matmul_shared(const float* A, const float* B, float* C, int M, int N, int K,
                   cudaStream_t stream = 0);

// With 1D tiling optimization
void matmul_1d_tiling(const float* A, const float* B, float* C, int M, int N, int K,
                      cudaStream_t stream = 0);

// With 2D tiling optimization
void matmul_2d_tiling(const float* A, const float* B, float* C, int M, int N, int K,
                      cudaStream_t stream = 0);

// Vectorized loads
void matmul_vectorized(const float* A, const float* B, float* C, int M, int N, int K,
                       cudaStream_t stream = 0);

// Best implementation
void matmul(const float* A, const float* B, float* C, int M, int N, int K,
            cudaStream_t stream = 0);

} // namespace matmul
