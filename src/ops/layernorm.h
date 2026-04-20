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

// ============================================
// Stage 1-3: Advanced Optimizations
// ============================================

namespace stage1 {
// 激进的向量化 + 循环展开
void layernorm_stage1(const float* input, const float* weight,
                     const float* bias, float* output, int rows, int cols,
                     float eps = 1e-5f, cudaStream_t stream = 0);
}  // namespace stage1

namespace stage2 {
// 在线算法（单遍遍历）
void layernorm_stage2_online(const float* input, const float* weight,
                            const float* bias, float* output, int rows, int cols,
                            float eps = 1e-5f, cudaStream_t stream = 0);

// Kahan 求和（减少精度损失）
void layernorm_stage2_kahan(const float* input, const float* weight,
                           const float* bias, float* output, int rows, int cols,
                           float eps = 1e-5f, cudaStream_t stream = 0);
}  // namespace stage2

namespace stage3 {
// 激进的 Warp 优化 + 寄存器缓存
void layernorm_stage3_aggressive(const float* input, const float* weight,
                                const float* bias, float* output, int rows, int cols,
                                float eps = 1e-5f, cudaStream_t stream = 0);

// 向量化 + ILP
void layernorm_stage3_vectorized_ilp(const float* input, const float* weight,
                                     const float* bias, float* output, int rows, int cols,
                                     float eps = 1e-5f, cudaStream_t stream = 0);

// 自动选择最优实现
void layernorm_stage3(const float* input, const float* weight,
                     const float* bias, float* output, int rows, int cols,
                     float eps = 1e-5f, cudaStream_t stream = 0);
}  // namespace stage3

} // namespace layernorm
