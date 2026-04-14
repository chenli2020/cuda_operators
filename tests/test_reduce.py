"""
Reduce Sum 测试与基准测试

学习目标：
1. 理解如何验证 CUDA 算子的数值正确性
2. 学习如何测量和分析算子性能
3. 掌握与 PyTorch 对比测试的方法
4. 理解带宽计算和 Roofline 模型

测试方法论：
1. 精度测试：与 PyTorch 结果对比，相对误差 < 1e-3 认为通过
2. 性能测试：测量执行时间和内存带宽
3. 不同规模测试：从小规模到大规模，验证算法扩展性
"""

import numpy as np
import torch
import sys
import os

# 尝试导入编译好的 CUDA 模块
try:
    import cuda_ops
except ImportError:
    # 如果失败，添加 build 目录到路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
    import cuda_ops

from utils import check_allclose, generate_random_input


def test_reduce_sum():
    """
    Reduce Sum 精度测试

    测试策略：
    1. 生成随机输入
    2. 用 PyTorch 计算参考结果
    3. 用不同 CUDA 实现计算结果
    4. 对比误差，确保在可接受范围内

    为什么需要测试所有实现：
    - naive、shared、warp 等不同实现可能有不同的数值误差
    - 需要确保优化不会破坏正确性
    """
    print("Testing reduce_sum...")

    # 测试不同规模的数据
    test_cases = [
        (100,),      # 小规模
        (1000,),     # 中规模
        (10000,),    # 较大规模
        (100000,),   # 大规模
    ]

    all_passed = True
    for shape in test_cases:
        input_np = generate_random_input(shape)
        input_torch = torch.from_numpy(input_np)

        # PyTorch 参考实现
        expected = input_torch.sum().item()

        # 测试所有 CUDA 实现
        for impl in ["naive", "shared", "warp", "twopass", "auto"]:
            try:
                actual = cuda_ops.reduce_sum(input_np, impl)[0]

                # 检查精度
                # rtol: 相对误差容忍度
                # atol: 绝对误差容忍度
                passed = check_allclose([actual], [expected],
                                        rtol=1e-4,
                                        atol=1e-5,
                                        name=f"shape={shape}, impl={impl}")
                all_passed = all_passed and passed
            except Exception as e:
                print(f"  Error with {impl}: {e}")
                all_passed = False

    return all_passed


def benchmark_reduce():
    """
    Reduce Sum 性能基准测试

    关键概念：
    - 内存带宽：每秒传输的数据量（GB/s）
    - 理论峰值：GPU 内存的理论最大带宽（如 A100 约 2 TB/s）
    - 带宽利用率：实际带宽 / 理论峰值，目标 > 80%

    Reduce 是内存带宽密集型操作：
    - 计算复杂度：O(n)
    - 内存访问：O(n) 读取 + O(1) 写入
    - 计算/内存比：极低，性能完全受限于内存带宽

    为什么测量带宽而不是时间：
    - 时间受数据规模影响，不便横向比较
    - 带宽可以直观看出是否达到硬件极限
    """
    print("\n" + "="*60)
    print("Reduce Sum Benchmark")
    print("="*60)

    import time

    # 测试不同规模的数据
    sizes = [10000, 100000, 1000000, 10000000]

    for n in sizes:
        input_np = generate_random_input((n,))
        input_torch = torch.from_numpy(input_np).cuda()

        # Warmup：预热 GPU，避免冷启动影响
        for _ in range(10):
            _ = cuda_ops.reduce_sum(input_np, "auto")
            _ = input_torch.sum()

        # 同步确保 warmup 完成
        torch.cuda.synchronize()

        # Benchmark CUDA 实现
        num_iters = 100
        start = time.time()
        for _ in range(num_iters):
            _ = cuda_ops.reduce_sum(input_np, "auto")
        cuda_time = (time.time() - start) / num_iters * 1000  # 转换为 ms

        # Benchmark PyTorch（cuDNN/cuBLAS）
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = input_torch.sum()
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / num_iters * 1000

        # 计算内存带宽
        # Reduce 需要读取所有输入（n 个 float）
        bytes_moved = n * 4  # float32 = 4 bytes
        cuda_bw = bytes_moved / (cuda_time / 1000) / 1e9  # GB/s
        torch_bw = bytes_moved / (torch_time / 1000) / 1e9

        print(f"  Size {n:>10}: CUDA={cuda_time:.3f}ms ({cuda_bw:.2f}GB/s), "
              f"PyTorch={torch_time:.3f}ms ({torch_bw:.2f}GB/s)")

        # 性能分析提示
        # A100 理论带宽约 2 TB/s = 2000 GB/s
        # 如果能达到 80% (1600 GB/s) 说明优化很好
        if cuda_bw > 1000:
            print(f"    [Good] Bandwidth utilization: {cuda_bw/2000*100:.1f}% (A100)")


if __name__ == "__main__":
    # 先运行精度测试
    passed = test_reduce_sum()

    # 再运行性能测试
    benchmark_reduce()

    if passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)
