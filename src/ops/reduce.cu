/**
 * Reduce Sum 算子实现 - 从 Naive 到优化的完整演进
 *
 * 学习目标：
 * 1. 理解为什么 atomicAdd 是性能瓶颈（全局内存争用）
 * 2. 学习使用共享内存进行 block 级别归约
 * 3. 掌握 warp shuffle 指令进行 warp 级别归约
 * 4. 理解两级归约策略（处理任意大小数组）
 *
 * 性能目标：达到设备内存带宽的 80%+
 */

#include "reduce.h"
#include "cuda_utils.h"
#include <cfloat>
#include <cmath>
#include <algorithm>

namespace reduce {

/**
 * Naive 实现 - 每个线程一个元素，直接 atomicAdd 到全局内存
 *
 * 问题分析：
 * - 所有线程同时写入同一个内存地址 output
 * - 导致严重的内存争用（memory contention）
 * - 虽然结果正确，但性能极差（带宽利用率 < 1%）
 *
 * 适用场景：仅作为 baseline，实际不使用
 */
__global__ void reduce_sum_naive_kernel(const float *input, float *output,
                                        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // atomicAdd: 原子操作，保证并发安全但性能差
        atomicAdd(output, input[idx]);
    }
}

/**
 * Shared Memory 优化 - 每个 block 先归约，再原子累加
 *
 * 优化思路：
 * 1. 每个 block 内部的线程先把数据加载到共享内存
 * 2. 在共享内存中进行树形归约（速度快，无全局内存争用）
 * 3. 每个 block 只产生一个部分和，再用 atomicAdd
 *
 * 效果：将 n 次 atomicAdd 减少到 grid_size 次
 */
template <int BLOCK_SIZE>
__global__ void reduce_sum_shared_kernel(const float *input, float *output,
                                         int n) {
    // 静态分配的共享内存，block 内所有线程可见
    __shared__ float shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 阶段 1：从全局内存加载数据到共享内存
    // 越界检查：如果 idx >= n，则贡献 0
    shared[tid] = (idx < n) ? input[idx] : 0.0f;

    // __syncthreads()：确保 block 内所有线程完成加载
    // 这是 CUDA 编程中最重要的同步原语之一
    __syncthreads();

    // 阶段 2：树形归约（in-place reduction）
    // 每次迭代，线程 tid 累加距离为 s 的元素
    // 例如 BLOCK_SIZE=256：
    //   s=128: tid 0-127 累加 shared[128-255]
    //   s=64:  tid 0-63  累加 shared[64-127]
    //   ...直到 s=1
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        // 必须同步，确保本轮归约完成后再进入下一轮
        __syncthreads();
    }

    // 阶段 3：block 的 sum 在 shared[0]，原子累加到全局结果
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

/**
 * Warp Shuffle 优化 - 使用寄存器级别的数据交换
 *
 * 关键洞察：
 * - 一个 warp（32 线程）在执行时是物理上同步的（SIMT）
 * - __shfl_down_sync 可以在 warp 内直接交换寄存器值，无需共享内存
 * - 延迟远低于 shared memory 访问
 *
 * 实现策略：
 * 1. 每个 warp 先用 warp_reduce_sum 归约（寄存器级别）
 * 2. 每个 warp 的第一个线程把部分和写入共享内存
 * 3. 第一个 warp 再归约这些部分和
 */
template <int BLOCK_SIZE>
__global__ void reduce_sum_warp_kernel(const float *input, float *output,
                                       int n) {
    int tid = threadIdx.x;

    // Grid-stride loop: 处理大数组时，一个线程处理多个元素
    // 策略：线程 idx 处理 idx, idx+grid_size, idx+2*grid_size, ...
    // 这样保持内存访问的合并（coalesced）模式
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // 累加该线程负责的所有元素
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Step 1: Warp 级别归约（使用 warp_shuffle）
    // warp_reduce_sum 内部使用 __shfl_down_sync
    sum = warp_reduce_sum(sum);

    // Step 2: 每个 warp 的线程 0 把结果写入共享内存
    // 最多 32 个 warps 每 block（1024/32 = 32）
    __shared__ float warp_sums[32];
    int warp_id = tid / 32;      // 当前线程属于哪个 warp
    int lane_id = tid % 32;      // 在 warp 内的位置（0-31）

    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Step 3: 第一个 warp 归约所有 warp 的部分和
    if (warp_id == 0) {
        // 加载本 warp 需要处理的数据
        sum = (lane_id < blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;

        // 再次 warp 归约
        sum = warp_reduce_sum(sum);

        // 最终结果写入全局内存
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

/**
 * 两级归约 - 处理任意大小的数组
 *
 * 问题：当数组非常大时，一个 block 处理不过来
 * 解决：
 *   第一级：每个 block 产生一个部分和，存储到中间数组
 *   第二级：再启动一个 kernel 归约这些部分和
 *
 * 优势：
 * - 可以处理任意大小的数组
 * - 第一级 kernel 完全并行，无原子操作争用
 */
template <int BLOCK_SIZE>
__global__ void reduce_sum_first_pass(const float *input, float *partial_sums,
                                      int n) {
    __shared__ float shared[BLOCK_SIZE];

    int tid = threadIdx.x;

    // Grid-stride loop 累加
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n;
         i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Block 级别归约
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    // 每个 block 写入一个部分和
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

// Host 函数实现

void reduce_sum_naive(const float *input, float *output, int n,
                      cudaStream_t stream) {
    // 清零输出，因为我们要累加进去
    CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float), stream));

    const int block_size = 256;
    const int grid_size = div_up(n, block_size);

    reduce_sum_naive_kernel<<<grid_size, block_size, 0, stream>>>(input,
                                                                   output, n);
    CUDA_CHECK(cudaGetLastError());
}

void reduce_sum_shared(const float *input, float *output, int n,
                       cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float), stream));

    const int block_size = 256;
    const int grid_size = div_up(n, block_size);

    reduce_sum_shared_kernel<256>
        <<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

void reduce_sum_warp(const float *input, float *output, int n,
                     cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float), stream));

    const int block_size = 256;
    // 限制 grid 大小，避免过多 block（经验值：最多 128 个 block）
    const int num_blocks = std::min(div_up(n, block_size), 128);

    reduce_sum_warp_kernel<256>
        <<<num_blocks, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

void reduce_sum_twopass(const float *input, float *output, int n,
                        cudaStream_t stream) {
    const int block_size = 256;
    // 第一级产生的部分和数量
    const int num_blocks = std::min(div_up(n, block_size), 128);

    // 分配临时存储
    float *partial_sums;
    CUDA_CHECK(cudaMalloc(&partial_sums, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(partial_sums, 0, num_blocks * sizeof(float),
                               stream));

    // 第一级归约
    reduce_sum_first_pass<256>
        <<<num_blocks, block_size, 0, stream>>>(input, partial_sums, n);

    // 第二级归约：一个 block 就够了
    CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float), stream));
    reduce_sum_shared_kernel<256>
        <<<1, block_size, 0, stream>>>(partial_sums, output, num_blocks);

    CUDA_CHECK(cudaFree(partial_sums));
    CUDA_CHECK(cudaGetLastError());
}

/**
 * 自动选择最优实现
 *
 * 启发式策略：
 * - 小数组 (< 1000)：shared memory 简单高效
 * - 中等数组 (< 1M)：warp shuffle，减少同步开销
 * - 大数组 (>= 1M)：two-pass，充分利用所有 SMs
 */
void reduce_sum(const float *input, float *output, int n,
                cudaStream_t stream) {
    if (n < 1000) {
        reduce_sum_shared(input, output, n, stream);
    } else if (n < 1000000) {
        reduce_sum_warp(input, output, n, stream);
    } else {
        reduce_sum_twopass(input, output, n, stream);
    }
}

} // namespace reduce
