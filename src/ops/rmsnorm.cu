/**
 * RMSNorm (Root Mean Square Layer Normalization) 算子实现
 *
 * 学习目标：
 * 1. 理解 RMSNorm 相比 LayerNorm 的优势（更简单、更快）
 * 2. 学习 LLaMA/Mistral 等现代 Transformer 的归一化选择
 * 3. 掌握 fused kernel 设计（add + RMSNorm 合并）
 *
 * RMSNorm 公式：
 *   rms = sqrt(mean(x^2) + eps)
 *   y = x / rms * weight
 *
 * 与 LayerNorm 的区别：
 * - LayerNorm: (x - mean) / sqrt(var + eps)  [需要计算 mean 和 var]
 * - RMSNorm:   x / sqrt(mean(x^2) + eps)      [只需计算 mean of squares]
 * - 省去了 mean 计算和减法操作
 * - 在 LLM 中效果相近但速度更快
 *
 * 来源论文: "Root Mean Square Layer Normalization" (2019)
 */

#include "rmsnorm.h"
#include "cuda_utils.h"
#include <cmath>
#include <algorithm>

namespace rmsnorm {

/**
 * Naive RMSNorm - 每个线程处理一整行
 *
 * 执行步骤：
 * 1. 计算 x^2 的平均值（RMS 的平方）
 * 2. 计算 inv_rms = 1 / sqrt(rms_sq + eps)
 * 3. 每个元素乘以 inv_rms * weight
 */
__global__ void rmsnorm_naive_kernel(const float* input, const float* weight,
                                     float* output, int rows, int cols,
                                     float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Pass 1: 计算 x^2 的平均值
    float sum_sq = 0.0f;
    for (int i = 0; i < cols; ++i) {
        sum_sq += in_row[i] * in_row[i];
    }
    float rms = sqrtf(sum_sq / cols + eps);
    float inv_rms = 1.0f / rms;

    // Pass 2: 归一化并应用 weight
    for (int i = 0; i < cols; ++i) {
        out_row[i] = in_row[i] * inv_rms * weight[i];
    }
}

/**
 * Warp-Level 优化 RMSNorm
 *
 * 与 LayerNorm 类似，但只需要计算一个统计量（mean of squares）
 * 所以比 LayerNorm 少一次 reduce 操作，速度更快
 */
template <int BLOCK_SIZE>
__global__ void rmsnorm_warp_kernel(const float* input, const float* weight,
                                    float* output, int rows, int cols,
                                    float eps) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Step 1: 并行计算 sum of squares
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = in_row[i];
        thread_sum_sq += val * val;
    }

    // Warp reduce sum
    thread_sum_sq = warp_reduce_sum(thread_sum_sq);

    // 收集并得到最终 inv_rms
    __shared__ float inv_rms;
    if (tid % 32 == 0) {
        // 使用共享内存的一部分存储 warp 部分和
        __shared__ float shared_sum_sq[BLOCK_SIZE / 32];
        shared_sum_sq[tid / 32] = thread_sum_sq;
    }
    __syncthreads();

    if (tid < 32) {
        __shared__ float* shared_sum_sq = reinterpret_cast<float*>(
            &inv_rms - BLOCK_SIZE / 32);
        thread_sum_sq = (tid < BLOCK_SIZE / 32) ? shared_sum_sq[tid] : 0.0f;
        thread_sum_sq = warp_reduce_sum(thread_sum_sq);
        if (tid == 0) {
            float mean_sq = thread_sum_sq / cols;
            inv_rms = 1.0f / sqrtf(mean_sq + eps);
        }
    }
    __syncthreads();

    // Step 2: 归一化并写入结果
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        out_row[i] = in_row[i] * inv_rms * weight[i];
    }
}

/**
 * Vectorized RMSNorm - 向量化加载优化
 *
 * 与 LayerNorm 的向量化版本相同，但计算更简单
 */
template <int BLOCK_SIZE>
__global__ void rmsnorm_vectorized_kernel(const float* input,
                                          const float* weight, float* output,
                                          int rows, int cols, float eps) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // 向量化指针转换
    const float4* in_vec = reinterpret_cast<const float4*>(in_row);
    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    float4* out_vec = reinterpret_cast<float4*>(out_row);
    int vec_cols = cols / 4;

    // Step 1: 计算 sum of squares（向量化）
    float4 thread_sum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = in_vec[i];
        thread_sum4.x += v.x * v.x;
        thread_sum4.y += v.y * v.y;
        thread_sum4.z += v.z * v.z;
        thread_sum4.w += v.w * v.w;
    }

    float thread_sum_sq = thread_sum4.x + thread_sum4.y + thread_sum4.z + thread_sum4.w;
    thread_sum_sq = warp_reduce_sum(thread_sum_sq);

    __shared__ float inv_rms;
    if (tid % 32 == 0) {
        __shared__ float shared_sum_sq[BLOCK_SIZE / 32];
        shared_sum_sq[tid / 32] = thread_sum_sq;
    }
    __syncthreads();

    if (tid < 32) {
        __shared__ float* shared_sum_sq = reinterpret_cast<float*>(
            &inv_rms - BLOCK_SIZE / 32);
        thread_sum_sq = (tid < BLOCK_SIZE / 32) ? shared_sum_sq[tid] : 0.0f;
        thread_sum_sq = warp_reduce_sum(thread_sum_sq);
        if (tid == 0) {
            float mean_sq = thread_sum_sq / cols;
            inv_rms = 1.0f / sqrtf(mean_sq + eps);
        }
    }
    __syncthreads();

    // Step 2: 归一化并写入（向量化）
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = in_vec[i];
        float4 w = w_vec[i];

        float4 out;
        out.x = v.x * inv_rms * w.x;
        out.y = v.y * inv_rms * w.y;
        out.z = v.z * inv_rms * w.z;
        out.w = v.w * inv_rms * w.w;

        out_vec[i] = out;
    }
}

/**
 * Fused Add + RMSNorm - 融合加法与归一化
 *
 * 常见场景：Transformer 中的残差连接
 *   output = RMSNorm(input + residual)
 *
 * 融合优势：
 * - 减少一次全局内存读取（input + residual 可以 on-the-fly 计算）
 * - 减少 kernel 启动开销
 * - 更好的内存局部性
 *
 * 这是生产代码中非常重要的优化技巧
 */
template <int BLOCK_SIZE>
__global__ void rmsnorm_fused_add_kernel(const float* input, const float* residual,
                                         const float* weight, float* output,
                                         int rows, int cols, float eps) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    const float* res_row = residual + row * cols;
    float* out_row = output + row * cols;

    // Step 1: 融合加法 + 计算 sum of squares
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = in_row[i] + res_row[i];  // 融合加法
        out_row[i] = val;  // 暂存到输出（避免再次读取）
        thread_sum_sq += val * val;
    }

    // Warp reduce
    thread_sum_sq = warp_reduce_sum(thread_sum_sq);

    __shared__ float inv_rms;
    if (tid % 32 == 0) {
        __shared__ float shared_sum_sq[BLOCK_SIZE / 32];
        shared_sum_sq[tid / 32] = thread_sum_sq;
    }
    __syncthreads();

    if (tid < 32) {
        __shared__ float* shared_sum_sq = reinterpret_cast<float*>(
            &inv_rms - BLOCK_SIZE / 32);
        thread_sum_sq = (tid < BLOCK_SIZE / 32) ? shared_sum_sq[tid] : 0.0f;
        thread_sum_sq = warp_reduce_sum(thread_sum_sq);
        if (tid == 0) {
            float mean_sq = thread_sum_sq / cols;
            inv_rms = 1.0f / sqrtf(mean_sq + eps);
        }
    }
    __syncthreads();

    // Step 2: 应用归一化（从输出读取，已经包含相加结果）
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        out_row[i] = out_row[i] * inv_rms * weight[i];
    }
}

// Host 实现

void rmsnorm_naive(const float* input, const float* weight, float* output,
                   int rows, int cols, float eps, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = div_up(rows, block_size);

    rmsnorm_naive_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, output, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_warp(const float* input, const float* weight, float* output,
                  int rows, int cols, float eps, cudaStream_t stream) {
    const int block_size = 256;

    rmsnorm_warp_kernel<256><<<rows, block_size, 0, stream>>>(
        input, weight, output, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_vectorized(const float* input, const float* weight, float* output,
                        int rows, int cols, float eps, cudaStream_t stream) {
    if (cols % 4 != 0) {
        rmsnorm_warp(input, weight, output, rows, cols, eps, stream);
        return;
    }

    const int block_size = 256;

    rmsnorm_vectorized_kernel<256><<<rows, block_size, 0, stream>>>(
        input, weight, output, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm(const float* input, const float* weight, float* output, int rows,
             int cols, float eps, cudaStream_t stream) {
    if (cols % 4 == 0 && cols >= 256) {
        rmsnorm_vectorized(input, weight, output, rows, cols, eps, stream);
    } else {
        rmsnorm_warp(input, weight, output, rows, cols, eps, stream);
    }
}

}  // namespace rmsnorm
