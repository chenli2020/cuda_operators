#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <cstring>

/**
 * CUDA 错误检查宏
 *
 * 使用方法：
 *   CUDA_CHECK(cudaMalloc(...));
 *   CUDA_CHECK(cudaMemcpy(...));
 *
 * 如果 CUDA 调用返回错误，会打印文件名和行号，并抛出异常
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

/**
 * Kernel 启动配置结构体
 *
 * grid:     grid 维度（多少个 block）
 * block:    block 维度（每个 block 多少线程）
 * shared_mem: 动态共享内存大小（字节）
 */
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem = 0;
};

/**
 * 获取 1D 操作的启动配置
 *
 * 策略：每个线程处理一个元素，grid 大小覆盖所有数据
 *
 * @param n           元素总数
 * @param block_size  每个 block 的线程数（默认 256，通常是 warp 大小 32 的倍数）
 * @return LaunchConfig 包含 grid 和 block 配置
 */
inline LaunchConfig get_launch_config_1d(int n, int block_size = 256) {
    LaunchConfig config;
    config.block = dim3(block_size);
    // 向上取整：(n + block_size - 1) / block_size
    config.grid = dim3((n + block_size - 1) / block_size);
    return config;
}

/**
 * 获取 2D 操作的启动配置（用于图像、矩阵等 2D 数据）
 *
 * @param h        高度（行数）
 * @param w        宽度（列数）
 * @param block_x  x 维度 block 大小
 * @param block_y  y 维度 block 大小
 */
inline LaunchConfig get_launch_config_2d(int h, int w, int block_x = 16,
                                          int block_y = 16) {
    LaunchConfig config;
    config.block = dim3(block_x, block_y);
    config.grid =
        dim3((w + block_x - 1) / block_x, (h + block_y - 1) / block_y);
    return config;
}

/**
 * Warp 级别归约 - 求和
 *
 * 原理：使用 __shfl_down_sync 在 warp 内进行线程间数据交换
 *       不需要共享内存，比 shared memory 方式更快
 *
 * 执行过程（以 warp 大小 32 为例，val 是每个线程持有的值）：
 *   offset=16: 线程 0-15 获取线程 16-31 的值，累加到自己
 *   offset=8:  线程 0-7 获取线程 8-15 的值，累加
 *   ...直到 offset=1
 *   最终线程 0 持有 warp 的总和
 *
 * @param val 每个线程输入的值
 * @return 整个 warp 的和（只有线程 0 的结果有效，或广播给所有线程）
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // 0xFFFFFFFF 表示所有 32 个线程都参与
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

/**
 * Warp 级别归约 - 求最大值
 *
 * 与 warp_reduce_sum 类似，只是操作换成 max
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

/**
 * Block 级别归约 - 求和（使用共享内存）
 *
 * 执行过程：
 * 1. 每个线程写入自己的值到共享内存
 * 2. 进行树形归约：每次将距离为 s 的两个元素相加
 * 3. 最终 shared[0] 存储 block 的总和
 *
 * @tparam BLOCK_SIZE block 大小（必须是 2 的幂）
 * @param val    当前线程的值
 * @param shared 共享内存数组，大小至少为 BLOCK_SIZE
 * @return block 的总和
 */
template <int BLOCK_SIZE>
__device__ __forceinline__ float
block_reduce_sum(float val, float *shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();  // 确保所有线程写入完成

    // 树形归约：从半长开始，每次减半
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();  // 等待本轮归约完成
    }

    return shared[0];
}

/**
 * Block 级别归约 - 求最大值
 */
template <int BLOCK_SIZE>
__device__ __forceinline__ float
block_reduce_max(float val, float *shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        __syncthreads();
    }

    return shared[0];
}

/**
 * 整数除法向上取整
 *
 * 用途：计算需要多少个 block 才能覆盖 n 个元素
 * 例如：div_up(1000, 256) = 4（需要 4 个 block，每个 256 线程）
 */
__host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}
