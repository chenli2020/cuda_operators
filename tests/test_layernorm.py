"""
LayerNorm 算子测试与基准测试

学习目标：
1. 理解如何验证 CUDA 算子的数值正确性
2. 学习如何测量和分析算子性能
3. 掌握与 PyTorch 对比测试的方法
4. 理解带宽计算和性能瓶颈分析

测试方法论：
1. 精度测试：与 PyTorch 结果对比，相对误差 < 1e-4 认为通过
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


def pytorch_layernorm(x, weight, bias, eps=1e-5):
    """
    PyTorch LayerNorm 参考实现

    作用：作为"金标准"验证 CUDA 实现的正确性

    参数：
        x: 输入数据，shape = (rows, cols)
        weight: 缩放参数，shape = (cols,)
        bias: 偏移参数，shape = (cols,)
        eps: 防止除零的小常数

    返回：
        LayerNorm 的输出

    为什么使用 PyTorch 作为参考：
    - PyTorch 的 LayerNorm 经过充分测试和优化
    - 数值稳定性好，被广泛使用
    - 提供了可靠的基准对比
    """
    # 将 numpy 数组转换为 PyTorch 张量
    t = torch.from_numpy(x)
    w = torch.from_numpy(weight)
    b = torch.from_numpy(bias)

    # 创建 LayerNorm 层
    # elementwise_affine=True: 应用 weight 和 bias
    # eps: 数值稳定性的 epsilon
    ln = torch.nn.LayerNorm(x.shape[-1], elementwise_affine=True, eps=eps)

    # 设置可训练参数
    ln.weight.data = w
    ln.bias.data = b

    # 前向传播（不需要梯度）
    with torch.no_grad():
        return ln(t).numpy()


def test_layernorm():
    """
    LayerNorm 精度测试

    测试策略：
    1. 生成随机输入数据
    2. 用 PyTorch 计算参考结果
    3. 用不同 CUDA 实现计算结果
    4. 对比误差，确保在可接受范围内

    为什么需要测试所有实现：
    - naive、warp、vectorized 等不同实现可能有不同的数值误差
    - 需要确保优化不会破坏正确性
    - 不同的计算顺序可能导致浮点精度差异
    """
    print("Testing layernorm...")

    # 测试不同规模的数据
    # (rows, cols): 行数和列数
    test_cases = [
        (2, 128),      # 极小规模：快速验证基本功能
        (32, 512),     # 小规模：测试基本性能
        (32, 1024),    # 中等规模：测试向量化优化（4的倍数）
        (128, 4096),   # 大规模：测试内存带宽利用
        (128, 8192),   # 超大规模：测试极限性能
    ]

    all_passed = True

    for rows, cols in test_cases:
        # 生成随机测试数据
        input_np = generate_random_input((rows, cols))

        # 初始化 weight 和 bias
        # weight=1, bias=0 相当于不做仿射变换
        weight_np = np.ones(cols, dtype=np.float32)
        bias_np = np.zeros(cols, dtype=np.float32)

        # 计算参考结果（PyTorch）
        expected = pytorch_layernorm(input_np, weight_np, bias_np)

        # 测试所有 CUDA 实现
        for impl in ["naive", "warp", "vectorized", "auto"]:
            try:
                # 调用 CUDA 实现
                # 返回值：actual 是 numpy 数组
                actual = cuda_ops.layernorm(input_np, weight_np, bias_np,
                                            rows, cols, 1e-5, impl)

                # 检查精度
                # rtol: 相对误差容忍度（相对误差 = |actual - expected| / |expected|）
                # atol: 绝对误差容忍度（绝对误差 = |actual - expected|）
                passed = check_allclose(actual, expected, rtol=1e-4, atol=1e-5,
                                        name=f"rows={rows}, cols={cols}, impl={impl}")
                all_passed = all_passed and passed

            except Exception as e:
                # 如果实现出错，打印错误信息
                print(f"  Error with {impl}: {e}")
                all_passed = False

    return all_passed


def benchmark_layernorm():
    """
    LayerNorm 性能基准测试

    关键概念：
    - 内存带宽：每秒传输的数据量（GB/s）
    - 理论峰值：GPU 内存的理论最大带宽（如 A100 约 2 TB/s）
    - 带宽利用率：实际带宽 / 理论峰值，目标 > 80%

    LayerNorm 是内存带宽密集型操作：
    - 计算复杂度：O(rows * cols)
    - 内存访问：O(rows * cols) 读取 + O(rows * cols) 写入
    - 计算/内存比：低，性能主要受限于内存带宽

    为什么测量带宽而不是时间：
    - 时间受数据规模影响，不便横向比较
    - 带宽可以直观看出是否达到硬件极限
    - 带宽利用率是优化的关键指标
    """
    print("\nBenchmarking layernorm...")

    import time

    # 测试不同规模的数据（比精度测试更大）
    test_cases = [
        (32, 512),      # 小规模
        (32, 1024),     # 中等规模
        (128, 4096),    # 大规模
        (256, 8192),    # 超大规模
    ]

    for rows, cols in test_cases:
        # 生成测试数据
        input_np = generate_random_input((rows, cols))
        weight_np = np.ones(cols, dtype=np.float32)
        bias_np = np.zeros(cols, dtype=np.float32)

        # 准备 GPU 数据（PyTorch）
        input_torch = torch.from_numpy(input_np).cuda()
        weight_torch = torch.from_numpy(weight_np).cuda()
        bias_torch = torch.from_numpy(bias_np).cuda()

        # 创建 PyTorch LayerNorm 层
        ln = torch.nn.LayerNorm(cols, elementwise_affine=True, eps=1e-5).cuda()
        ln.weight.data = weight_torch
        ln.bias.data = bias_torch

        # ========================================
        # Warmup：预热 GPU
        # ========================================
        # 目的：
        # 1. 避免 GPU 冷启动影响测试结果
        # 2. 让 GPU 达到稳定工作状态
        # 3. 编译和初始化 CUDA kernels
        for _ in range(10):
            _ = cuda_ops.layernorm(input_np, weight_np, bias_np, rows, cols, 1e-5, "auto")
            _ = ln(input_torch)

        # 同步确保 warmup 完成
        torch.cuda.synchronize()

        # ========================================
        # Benchmark CUDA 实现
        # ========================================
        num_iters = 100  # 迭代次数，取平均减少误差

        # 开始计时
        start = time.time()
        for _ in range(num_iters):
            _ = cuda_ops.layernorm(input_np, weight_np, bias_np, rows, cols, 1e-5, "auto")
        cuda_time = (time.time() - start) / num_iters * 1000  # 转换为毫秒

        # ========================================
        # Benchmark PyTorch 实现（cuDNN）
        # ========================================
        torch.cuda.synchronize()  # 确保之前的操作完成

        start = time.time()
        for _ in range(num_iters):
            _ = ln(input_torch)
        torch.cuda.synchronize()  # 等待 GPU 完成
        torch_time = (time.time() - start) / num_iters * 1000

        # ========================================
        # 计算内存带宽
        # ========================================
        bytes_per_elem = 4  # float32 = 4 bytes

        # 计算总的数据传输量
        # 读取：
        #   - input: rows * cols 个元素
        #   - weight: cols 个元素
        #   - bias: cols 个元素
        # 写入：
        #   - output: rows * cols 个元素
        # 注意：这是简化的计算，实际还有中间计算和多次遍历
        total_bytes = (rows * cols * 3 + cols * 2) * bytes_per_elem

        # 计算带宽：数据量 / 时间
        cuda_bw = total_bytes / (cuda_time / 1000) / 1e9  # GB/s
        torch_bw = total_bytes / (torch_time / 1000) / 1e9

        # 输出结果
        # 格式：[rows, cols]: CUDA时间 (CUDA带宽), PyTorch时间 (PyTorch带宽)
        print(f"  [{rows:4}, {cols:4}]: CUDA={cuda_time:.3f}ms ({cuda_bw:.2f}GB/s), "
              f"PyTorch={torch_time:.3f}ms ({torch_bw:.2f}GB/s)")

        # 性能分析提示
        # A100 理论带宽约 2 TB/s = 2000 GB/s
        # 如果能达到 80% (1600 GB/s) 说明优化很好
        # 但对于内存密集型操作，通常只能达到 30-50%


if __name__ == "__main__":
    """
    主函数：运行所有测试

    执行顺序：
    1. 先运行精度测试：确保实现正确
    2. 再运行性能测试：优化性能才有意义
    """
    # 精度测试
    passed = test_layernorm()

    # 性能测试
    benchmark_layernorm()

    # 根据测试结果退出
    if passed:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
