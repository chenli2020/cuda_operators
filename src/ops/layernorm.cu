/**
 * LayerNorm 算子实现 - 向量加载优化与并行统计
 *
 * 学习目标：
 * 1. 理解 LayerNorm 的计算流程（mean、variance、normalize、scale、shift）
 * 2. 掌握 float4 向量化内存访问（4x 带宽提升）
 * 3. 学习如何在 warp 内并行计算两个统计量
 * 4. 理解内存对齐对向量化加载的要求
 *
 * LayerNorm 公式：
 *   mean = sum(x) / N
 *   var = sum((x - mean)^2) / N
 *   y = (x - mean) / sqrt(var + eps) * weight + bias
 *
 * 注意：与 BatchNorm 不同，LayerNorm 是在特征维度上归一化
 */

#include "layernorm.h"
#include "cuda_utils.h"
#include <cmath>

namespace layernorm {

/**
 * Naive LayerNorm - 每个线程处理一整行
 *
 * 执行步骤：
 * 1. 计算均值（第一遍遍历）
 * 2. 计算方差（第二遍遍历）
 * 3. 归一化并应用权重和偏置（第三遍遍历）
 *
 * 问题：3 遍遍历数据，内存带宽利用率低
 */
__global__ void layernorm_naive_kernel(const float* input, const float* weight,
                                       const float* bias, float* output,
                                       int rows, int cols, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Pass 1: 计算均值
    float mean = 0.0f;
    for (int i = 0; i < cols; ++i) {
        mean += in_row[i];
    }
    mean /= cols;

    // Pass 2: 计算方差
    float var = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float diff = in_row[i] - mean;
        var += diff * diff;
    }
    var /= cols;

    // Pass 3: 归一化并应用 affine 变换
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < cols; ++i) {
        float normalized = (in_row[i] - mean) * inv_std;
        out_row[i] = normalized * weight[i] + bias[i];
    }
}

/**
 * Warp-Level 优化 LayerNorm
 *
 * 优化策略：
 * - 一个 block 处理一行（与 Softmax 类似）
 * - block 内线程并行处理列
 * - 使用 warp shuffle 计算 mean 和 var
 *
 * 减少遍历次数的方法：
 * - 可以同时计算 mean 和 sum of squares
 * - 但需要注意数值稳定性（Welford 算法）
 * 这里为了简单，仍然用两遍，但并行度更高
 */
template <int BLOCK_SIZE>
__global__ void layernorm_warp_kernel(const float* input, const float* weight,
                                      const float* bias, float* output,
                                      int rows, int cols, float eps) {
    // 共享内存用于 warp 间通信
    __shared__ float shared_mean[BLOCK_SIZE];
    __shared__ float shared_var[BLOCK_SIZE];

    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Step 1: 并行计算均值
    float thread_sum = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        thread_sum += in_row[i];
    }

    // Warp reduce sum
    thread_sum = warp_reduce_sum(thread_sum);

    // 收集所有 warp 的部分和
    __shared__ float mean;
    if (tid % 32 == 0) {
        shared_mean[tid / 32] = thread_sum;
    }
    __syncthreads();

    // 第一个 warp 归约得到最终 mean
    if (tid < 32) {
        thread_sum = (tid < BLOCK_SIZE / 32) ? shared_mean[tid] : 0.0f;
        thread_sum = warp_reduce_sum(thread_sum);
        if (tid == 0) mean = thread_sum / cols;
    }
    __syncthreads();

    // Step 2: 并行计算方差
    float thread_sq_diff = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float diff = in_row[i] - mean;
        thread_sq_diff += diff * diff;
    }

    // Warp reduce
    thread_sq_diff = warp_reduce_sum(thread_sq_diff);

    // 收集并得到最终 var 和 inv_std
    __shared__ float inv_std;
    if (tid % 32 == 0) {
        shared_var[tid / 32] = thread_sq_diff;
    }
    __syncthreads();

    if (tid < 32) {
        thread_sq_diff = (tid < BLOCK_SIZE / 32) ? shared_var[tid] : 0.0f;
        thread_sq_diff = warp_reduce_sum(thread_sq_diff);
        if (tid == 0) {
            float var = thread_sq_diff / cols;
            inv_std = 1.0f / sqrtf(var + eps);
        }
    }
    __syncthreads();

    // Step 3: 并行归一化并写入结果
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float normalized = (in_row[i] - mean) * inv_std;
        out_row[i] = normalized * weight[i] + bias[i];
    }
}

/**
 * Vectorized LayerNorm - 使用 float4 向量化加载
 *
 * 核心优化：
 * - 将 4 个 float 的加载/存储合并为一个 float4 操作
 * - 内存带宽利用率提升 4x（如果原始带宽是瓶颈）
 *
 * 限制：
 * - cols 必须是 4 的倍数（内存对齐）
 * - weight 和 bias 也必须对齐
 *
 * 实现细节：
 * - reinterpret_cast 将 float* 转为 float4*
 * - make_float4 构造向量
 * - 需要对齐的数据布局
 */
template <int BLOCK_SIZE>
__global__ void layernorm_vectorized_kernel(const float* input,
                                            const float* weight,
                                            const float* bias, float* output,
                                            int rows, int cols, float eps) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // 将指针转为 float4*，实现向量化访问
    // 假设 cols 是 4 的倍数，且数据 16 字节对齐
    const float4* in_vec = reinterpret_cast<const float4*>(in_row);
    float4* out_vec = reinterpret_cast<float4*>(out_row);
    int vec_cols = cols / 4;  // 向量元素个数是标量的 1/4

    // Step 1: 计算均值（向量化累加）
    // 每个 float4 包含 4 个 float，分别累加
    float4 thread_sum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = in_vec[i];
        thread_sum4.x += v.x;
        thread_sum4.y += v.y;
        thread_sum4.z += v.z;
        thread_sum4.w += v.w;
    }

    // 将 4 个分量的和合并
    float thread_sum = thread_sum4.x + thread_sum4.y + thread_sum4.z + thread_sum4.w;
    thread_sum = warp_reduce_sum(thread_sum);

    __shared__ float mean;
    __shared__ float inv_std;

    // 使用共享内存暂存 warp 部分和（与之前相同）
    if (tid % 32 == 0) {
        float* shared_sum = reinterpret_cast<float*>(
            &mean - BLOCK_SIZE / 32);
        shared_sum[tid / 32] = thread_sum;
    }
    __syncthreads();

    if (tid < 32) {
        float* shared_sum = reinterpret_cast<float*>(
            &mean - BLOCK_SIZE / 32);
        thread_sum = (tid < BLOCK_SIZE / 32) ? shared_sum[tid] : 0.0f;
        thread_sum = warp_reduce_sum(thread_sum);
        if (tid == 0) mean = thread_sum / cols;
    }
    __syncthreads();

    // Step 2: 计算方差（向量化）
    float4 thread_sq_diff4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = in_vec[i];
        // 计算每个分量与 mean 的差，然后平方
        float4 diff = make_float4(v.x - mean, v.y - mean, v.z - mean, v.w - mean);
        thread_sq_diff4.x += diff.x * diff.x;
        thread_sq_diff4.y += diff.y * diff.y;
        thread_sq_diff4.z += diff.z * diff.z;
        thread_sq_diff4.w += diff.w * diff.w;
    }

    float thread_sq_diff = thread_sq_diff4.x + thread_sq_diff4.y +
                           thread_sq_diff4.z + thread_sq_diff4.w;
    thread_sq_diff = warp_reduce_sum(thread_sq_diff);

    if (tid % 32 == 0) {
        float* shared_var = reinterpret_cast<float*>(
            &inv_std - BLOCK_SIZE / 32);
        shared_var[tid / 32] = thread_sq_diff;
    }
    __syncthreads();

    if (tid < 32) {
        float* shared_var = reinterpret_cast<float*>(
            &inv_std - BLOCK_SIZE / 32);
        thread_sq_diff = (tid < BLOCK_SIZE / 32) ? shared_var[tid] : 0.0f;
        thread_sq_diff = warp_reduce_sum(thread_sq_diff);
        if (tid == 0) {
            float var = thread_sq_diff / cols;
            inv_std = 1.0f / sqrtf(var + eps);
        }
    }
    __syncthreads();

    // Step 3: 归一化、应用 weight/bias，并向量化存储
    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    const float4* b_vec = reinterpret_cast<const float4*>(bias);

    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = in_vec[i];
        float4 w = w_vec[i];
        float4 b = b_vec[i];

        // 逐分量计算
        float4 out;
        out.x = ((v.x - mean) * inv_std) * w.x + b.x;
        out.y = ((v.y - mean) * inv_std) * w.y + b.y;
        out.z = ((v.z - mean) * inv_std) * w.z + b.z;
        out.w = ((v.w - mean) * inv_std) * w.w + b.w;

        out_vec[i] = out;  // 向量化存储
    }
}

// Host 实现

void layernorm_naive(const float* input, const float* weight,
                     const float* bias, float* output, int rows, int cols,
                     float eps, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = div_up(rows, block_size);

    layernorm_naive_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, bias, output, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void layernorm_warp(const float* input, const float* weight, const float* bias,
                    float* output, int rows, int cols, float eps,
                    cudaStream_t stream) {
    const int block_size = 256;

    layernorm_warp_kernel<256><<<rows, block_size, 0, stream>>>(
        input, weight, bias, output, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void layernorm_vectorized(const float* input, const float* weight,
                          const float* bias, float* output, int rows, int cols,
                          float eps, cudaStream_t stream) {
    // 只有 cols 是 4 的倍数时才使用向量化版本
    if (cols % 4 != 0) {
        layernorm_warp(input, weight, bias, output, rows, cols, eps, stream);
        return;
    }

    const int block_size = 256;

    layernorm_vectorized_kernel<256><<<rows, block_size, 0, stream>>>(
        input, weight, bias, output, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * 自动选择最优实现
 *
 * 启发式：
 * - cols >= 256 且是 4 的倍数：使用向量化版本（最大化带宽）
 * - 其他情况：使用 warp 版本（通用性好）
 */
void layernorm(const float* input, const float* weight, const float* bias,
               float* output, int rows, int cols, float eps,
               cudaStream_t stream) {
    if (cols % 4 == 0 && cols >= 256) {
        layernorm_vectorized(input, weight, bias, output, rows, cols, eps,
                             stream);
    } else {
        layernorm_warp(input, weight, bias, output, rows, cols, eps, stream);
    }
}

}  // namespace layernorm
