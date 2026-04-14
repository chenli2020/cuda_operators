/**
 * Softmax 算子实现 - 数值稳定性与并行优化
 *
 * 学习目标：
 * 1. 理解 Softmax 的数值稳定性问题（指数爆炸）
 * 2. 掌握 Online Softmax 算法（单遍计算）
 * 3. 学习如何在 warp 级别并行处理一行数据
 * 4. 理解内存访问模式对性能的影响
 *
 * Softmax 公式：softmax(x_i) = exp(x_i) / sum(exp(x_j))
 *
 * 数值稳定性技巧：
 *   令 m = max(x)，则 exp(x_i - m) / sum(exp(x_j - m))
 *   这样指数的最大值是 exp(0) = 1，不会溢出
 */

#include "softmax.h"
#include "cuda_utils.h"
#include <cmath>
#include <algorithm>

namespace softmax {

/**
 * Naive Softmax - 每个线程处理一整行
 *
 * 每个线程执行：
 * 1. 找到行的最大值（用于数值稳定性）
 * 2. 计算所有 exp(x_i - max)
 * 3. 累加得到分母
 * 4. 每个元素除以分母
 *
 * 问题：
 * - 当列数很大时，一个线程做太多工作
 * - 没有利用 warp 并行
 * - 需要 3 遍遍历数据（找 max、求 exp、归一化）
 */
__global__ void softmax_naive_kernel(const float* input, float* output,
                                     int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Pass 1: 找最大值（数值稳定性）
    float max_val = in_row[0];
    for (int i = 1; i < cols; ++i) {
        max_val = max(max_val, in_row[i]);
    }

    // Pass 2: 计算 exp 和总和
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float exp_val = expf(in_row[i] - max_val);
        out_row[i] = exp_val;  // 暂存 exp 值
        sum += exp_val;
    }

    // Pass 3: 归一化
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < cols; ++i) {
        out_row[i] *= inv_sum;
    }
}

/**
 * Online Softmax - 单遍算法（Streaming Softmax）
 *
 * 核心思想：不需要先找到全局最大值，可以在遍历中动态更新
 *
 * 算法推导：
 * 假设当前已处理的数据最大值为 m，总和为 d = sum(exp(x_i - m))
 * 新来一个值 x：
 *   - 如果 x <= m：直接累加 exp(x - m) 到 d
 *   - 如果 x > m：需要更新，新的 d = d * exp(m - x) + 1
 *     解释：把所有之前的 exp(x_i - m) 转换为以新 m 为基准
 *
 * 优势：只需遍历数据一次，减少内存访问
 */
__global__ void softmax_online_kernel(const float* input, float* output,
                                      int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Online softmax 变量
    float max_val = -INFINITY;  // 当前最大值
    float sum = 0.0f;           // 当前总和（相对于 max_val）

    // 单遍遍历
    for (int i = 0; i < cols; ++i) {
        float x = in_row[i];
        if (x > max_val) {
            // 发现新的最大值，需要调整之前累积的和
            // 原来的和是相对于旧 max_val 的
            // 需要乘以 exp(旧max - 新max) 来转换
            sum *= expf(max_val - x);
            max_val = x;
        }
        sum += expf(x - max_val);
    }

    // 第二遍：计算最终输出
    // 现在 max_val 是全局最大值，sum 是正确的总和
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < cols; ++i) {
        out_row[i] = expf(in_row[i] - max_val) * inv_sum;
    }
}

/**
 * Warp-Level Parallel Softmax - 多线程协作处理一行
 *
 * 突破：不再是一个线程处理一行，而是多个线程协作
 *
 * 策略：
 * - 每个 block 处理一行
 * - block 内的线程并行处理列（线程 tid 处理列 tid, tid+BLOCK_SIZE, ...）
 * - 使用 warp shuffle 找全局 max 和 sum
 *
 * 优势：
 * - 当 cols 很大时，并行度更高
 * - 更好的内存合并访问
 */
template <int BLOCK_SIZE>
__global__ void softmax_warp_kernel(const float* input, float* output,
                                    int rows, int cols) {
    // 共享内存用于 warp 之间的数据交换
    __shared__ float shared_max[BLOCK_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE];

    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Step 1: 并行找最大值
    // 每个线程处理自己负责的列的子集
    float thread_max = -INFINITY;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        thread_max = max(thread_max, in_row[i]);
    }

    // Warp reduce 找 max
    thread_max = warp_reduce_max(thread_max);

    // 第一个线程把本 warp 的 max 存入共享内存
    __shared__ float block_max;
    if (tid % 32 == 0) {
        shared_max[tid / 32] = thread_max;
    }
    __syncthreads();

    // 第一个 warp 归约所有 warp 的 max
    if (tid < 32) {
        thread_max = (tid < BLOCK_SIZE / 32) ? shared_max[tid] : -INFINITY;
        thread_max = warp_reduce_max(thread_max);
        if (tid == 0) block_max = thread_max;
    }
    __syncthreads();

    // Step 2: 并行计算 exp 和 sum
    float thread_sum = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float exp_val = expf(in_row[i] - block_max);
        out_row[i] = exp_val;  // 暂存
        thread_sum += exp_val;
    }

    // Warp reduce sum
    thread_sum = warp_reduce_sum(thread_sum);

    // 第一个线程把本 warp 的 sum 存入共享内存
    __shared__ float block_sum;
    if (tid % 32 == 0) {
        shared_sum[tid / 32] = thread_sum;
    }
    __syncthreads();

    // 第一个 warp 归约所有 warp 的 sum
    if (tid < 32) {
        thread_sum = (tid < BLOCK_SIZE / 32) ? shared_sum[tid] : 0.0f;
        thread_sum = warp_reduce_sum(thread_sum);
        if (tid == 0) block_sum = thread_sum;
    }
    __syncthreads();

    // Step 3: 并行归一化
    float inv_sum = 1.0f / block_sum;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        out_row[i] *= inv_sum;
    }
}

// Host 实现

void softmax_naive(const float* input, float* output, int rows, int cols,
                   cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = div_up(rows, block_size);

    softmax_naive_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void softmax_online(const float* input, float* output, int rows, int cols,
                    cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = div_up(rows, block_size);

    softmax_online_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void softmax_warp(const float* input, float* output, int rows, int cols,
                  cudaStream_t stream) {
    const int block_size = 256;

    softmax_warp_kernel<256><<<rows, block_size, 0, stream>>>(
        input, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * 自动选择最优实现
 *
 * 启发式：
 * - 列数 <= 1024：使用 warp 并行版本（充分利用并行度）
 * - 列数 > 1024：使用 online 版本（内存带宽更高效）
 */
void softmax(const float* input, float* output, int rows, int cols,
             cudaStream_t stream) {
    if (cols <= 1024) {
        softmax_warp(input, output, rows, cols, stream);
    } else {
        softmax_online(input, output, rows, cols, stream);
    }
}

}  // namespace softmax
