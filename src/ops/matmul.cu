/**
 * Matrix Multiplication (GEMM) 算子实现 - 内存优化到计算优化的演进
 *
 * 学习目标：
 * 1. 理解为什么 naive matmul 是内存带宽瓶颈（O(n^3) 计算 vs O(n^2) 内存）
 * 2. 掌握共享内存 tiling 技术（减少全局内存访问）
 * 3. 理解 1D/2D tiling 的区别（A 重用 vs A+B 重用）
 * 4. 学习如何计算理论峰值性能（GFLOPS）
 *
 * Matmul 公式：C[i,j] = sum_k(A[i,k] * B[k,j])
 *
 * 复杂度分析：
 * - 计算量：2 * M * N * K FLOPs（乘和加）
 * - 内存访问：读取 A (M*K) + B (K*N)，写入 C (M*N)
 * - 计算/访存比：O(K) - 当 K 很大时，计算密集型
 *
 * 优化路径：
 * Naive -> Shared Memory -> 1D Tiling -> 2D Tiling -> Register Tiling
 */

#include "matmul.h"
#include "cuda_utils.h"
#include <cstdio>

namespace matmul {

/**
 * Naive MatMul - 每个线程计算 C 的一个元素
 *
 * 内存访问模式分析：
 * - A[i,k]: 行优先，连续访问（好）
 * - B[k,j]: 列优先，stride = N（坏，非合并访问）
 * - 每个线程做 K 次内存读取，总共 M*N*K 次读取
 *
 * 性能瓶颈：
 * - B 的访问是 cache unfriendly 的
 * - 没有利用数据重用（A 的行和 B 的列被重复读取）
 */
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * Shared Memory Tiling - 核心优化技术
 *
 * 核心思想：
 * - 将 A 和 B 分成小块（tile），加载到共享内存
 * - 每个 block 计算 C 的一个 tile
 * - 在共享内存中重复利用数据，减少全局内存访问
 *
 * 内存访问优化：
 * - 原始：每个 A 元素被读取 N 次（每列 C 都用到）
 * - Tiling：每个 A 元素被加载到 shared mem 一次，被 block 内所有线程重用
 * - 全局内存访问减少到原来的 TILE_SIZE 分之一
 *
 * 执行流程（以 TILE_SIZE=32 为例）：
 * 1. 将 A 的 32x32 tile 从全局内存加载到共享内存
 * 2. 将 B 的 32x32 tile 从全局内存加载到共享内存
 * 3. 在这个 tile 上计算部分积
 * 4. 移动到下一个 tile，重复 1-3
 * 5. 累加所有 tile 的部分积得到最终结果
 */
const int TILE_SIZE = 32;

__global__ void matmul_shared_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    // 静态共享内存声明，block 内所有线程共享
    // As: A 的 tile，Bs: B 的 tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 计算该线程负责的 C 元素位置
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // 遍历所有 tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // 协作加载 A tile 到共享内存
        // 每个线程加载 A 的一个元素
        // 注意边界检查
        if (row < M && tile * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;  // padding
        }

        // 协作加载 B tile 到共享内存
        if (tile * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // 必须同步，确保 tile 加载完成才能计算
        __syncthreads();

        // 在共享内存上计算部分积
        // 此时 As 和 Bs 都在 shared memory，访问速度极快
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 必须同步，确保所有线程用完 tile 后才能加载下一个
        __syncthreads();
    }

    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * 1D Tiling - 进一步优化 A 的数据重用
 *
 * 洞察：
 * - 在 shared memory 版本中，A 的每一行被同列的 32 个线程读取
 * - 可以让每个线程计算 C 的多个元素（同一行的连续列）
 * - 这样 A 的行只加载一次，被多个计算重用
 *
 * 实现：
 * - 每个线程计算 C[row, col:col+4]（4 个连续元素）
 * - 使用 float4 向量化加载/存储
 */
const int TILE_K = 32;
const int TILE_N = 128;  // 每个线程计算 4 个元素，所以 tile 宽度是 4 的倍数

__global__ void matmul_1d_tiling_kernel(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
    // A tile: [TILE_K] - 每个 block 共享 A 的一列 tile
    // B tile: [TILE_K][TILE_N] - B 的 tile
    __shared__ float As[TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 每个线程处理一行，计算 TILE_N/4 个元素
    int row = by * blockDim.y + ty;
    int col_start = bx * TILE_N + tx * 4;

    // 每个线程维护自己的累加器
    float sum[4] = {0.0f};

    // 遍历 K 维度
    for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        // 加载 B tile（向量化，coalesced）
        // 每个线程加载 4 个元素
        for (int i = 0; i < 4; ++i) {
            int col = col_start + i;
            if (tile_k + ty < K && col < N) {
                Bs[ty][tx * 4 + i] = B[(tile_k + ty) * N + col];
            } else {
                Bs[ty][tx * 4 + i] = 0.0f;
            }
        }

        __syncthreads();

        // 计算：A 的行 × B 的 tile
        if (row < M) {
            for (int k = 0; k < TILE_K && tile_k + k < K; ++k) {
                float a = A[row * K + tile_k + k];
                // 一个 A 元素被用于计算 4 个 C 元素
                for (int i = 0; i < 4; ++i) {
                    sum[i] += a * Bs[k][tx * 4 + i];
                }
            }
        }

        __syncthreads();
    }

    // 写入结果
    if (row < M) {
        for (int i = 0; i < 4; ++i) {
            int col = col_start + i;
            if (col < N) {
                C[row * N + col] = sum[i];
            }
        }
    }
}

/**
 * 2D Tiling - 同时优化 A 和 B 的数据重用（最实用的优化）
 *
 * 洞察：
 * - 1D tiling 优化了 A 的重用，但 B 的每一列仍然只被用一次
 * - 可以让每个线程计算 C 的一个子矩阵（tile）
 * - 这样 A 的 tile 和 B 的 tile 都被充分重用
 *
 * 实现策略：
 * - 每个 block 计算 C 的 [BM][BN] 子矩阵
 * - block 内的线程布局为 [TM][TN]，每个线程计算 (BM/TM) × (BN/TN) 个元素
 * - A tile: [BM][BK], B tile: [BK][BN]
 */
const int BLOCK_M = 64;  // C tile 高度
const int BLOCK_N = 64;  // C tile 宽度
const int BLOCK_K = 16;  // K 维度 tile 大小

__global__ void matmul_2d_simple_kernel(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
    // 共享内存 tiles
    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Bs[BLOCK_K][BLOCK_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 该线程负责的 C 元素
    int row = by * BLOCK_M + ty;
    int col = bx * BLOCK_N + tx;

    float sum = 0.0f;

    // 遍历 K tiles
    for (int tile = 0; tile < (K + BLOCK_K - 1) / BLOCK_K; ++tile) {
        // 协作加载 A tile
        // 每个线程加载一个元素，需要循环直到整个 tile 加载完成
        if (row < M && tile * BLOCK_K + tx < K) {
            As[ty][tx] = A[row * K + tile * BLOCK_K + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // 协作加载 B tile
        if (tile * BLOCK_K + ty < K && col < N) {
            Bs[ty][tx] = B[(tile * BLOCK_K + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 计算部分积
        for (int k = 0; k < BLOCK_K; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host 实现

void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K,
                  cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    matmul_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void matmul_shared(const float* A, const float* B, float* C, int M, int N, int K,
                   cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_shared_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void matmul_1d_tiling(const float* A, const float* B, float* C, int M, int N, int K,
                      cudaStream_t stream) {
    dim3 block(TILE_N / 4, 1);
    dim3 grid((N + TILE_N - 1) / TILE_N, M);

    matmul_1d_tiling_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void matmul_2d_tiling(const float* A, const float* B, float* C, int M, int N, int K,
                      cudaStream_t stream) {
    dim3 block(BLOCK_N, BLOCK_M);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    matmul_2d_simple_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void matmul_vectorized(const float* A, const float* B, float* C, int M, int N, int K,
                       cudaStream_t stream) {
    // 使用 2D tiling 作为向量化版本
    matmul_2d_tiling(A, B, C, M, N, K, stream);
}

/**
 * 自动选择最优实现
 *
 * 启发式：
 * - 小矩阵：shared memory 版本（启动开销小）
 * - 大矩阵：2D tiling（充分利用数据重用）
 */
void matmul(const float* A, const float* B, float* C, int M, int N, int K,
            cudaStream_t stream) {
    if (M <= 32 || N <= 32 || K <= 32) {
        matmul_shared(A, B, C, M, N, K, stream);
    } else if (M >= 256 && N >= 256) {
        matmul_2d_tiling(A, B, C, M, N, K, stream);
    } else {
        matmul_shared(A, B, C, M, N, K, stream);
    }
}

}  // namespace matmul
