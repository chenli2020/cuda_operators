"""
LayerNorm 性能基准测试
对比不同优化阶段的性能表现

运行方式:
    python benchmark/benchmark_layernorm.py
"""

import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cuda_ops
except ImportError:
    print("错误: 无法导入 cuda_ops 模块")
    print("请先编译项目: python build_with_cmake.sh 或 build_with_cmake.bat")
    sys.exit(1)


def benchmark_layernorm(rows, cols, impl, num_iters=100, warmup=10):
    """
    测试单个实现的性能

    Args:
        rows, cols: 矩阵形状
        impl: 实现名称
        num_iters: 测试迭代次数
        warmup: 预热迭代次数

    Returns:
        平均时间 (ms), 带宽 (GB/s)
    """
    # 准备数据
    input = np.random.randn(rows, cols).astype(np.float32)
    weight = np.random.randn(cols).astype(np.float32)
    bias = np.random.randn(cols).astype(np.float32)

    # 预热
    for _ in range(warmup):
        _ = cuda_ops.layernorm(input, weight, bias, rows, cols, 1e-5f, impl)

    # 正式测试
    start = time.time()
    for _ in range(num_iters):
        output = cuda_ops.layernorm(input, weight, bias, rows, cols, 1e-5f, impl)
    end = time.time()

    avg_time_ms = (end - start) / num_iters * 1000

    # 计算带宽
    # 读取: input (rows*cols) + weight (cols) + bias (cols)
    # 写入: output (rows*cols)
    total_bytes = (rows * cols * 4 + cols * 4 + cols * 4 + rows * cols * 4)
    bandwidth_gb_s = total_bytes / (avg_time_ms / 1000) / 1e9

    return avg_time_ms, bandwidth_gb_s


def main():
    print("=" * 80)
    print("LayerNorm 性能基准测试")
    print("=" * 80)
    print()

    # 测试配置
    configs = [
        # (rows, cols, description)
        (128, 128, "小矩阵 (128x128)"),
        (128, 512, "中等矩阵 (128x512)"),
        (128, 1024, "中等矩阵 (128x1024)"),
        (128, 2048, "大矩阵 (128x2048)"),
        (128, 4096, "大矩阵 (128x4096)"),
        (4096, 4096, "超大矩阵 (4096x4096)"),
    ]

    # 测试的实现
    implementations = [
        ("naive", "Naive (3遍遍历)"),
        ("warp", "Warp 优化"),
        ("vectorized", "向量化 (float4)"),
        ("stage1", "Stage 1: 循环展开+Float8"),
        ("stage2_online", "Stage 2: 在线算法"),
        ("stage3", "Stage 3: 激进优化"),
        ("stage3_ilp", "Stage 3: 向量化+ILP"),
    ]

    results = {}

    for rows, cols, desc in configs:
        print(f"\n{'─' * 80}")
        print(f"测试: {desc} - rows={rows}, cols={cols}")
        print(f"{'─' * 80}")

        config_key = f"{rows}x{cols}"
        results[config_key] = {}

        for impl, impl_name in implementations:
            try:
                avg_time, bandwidth = benchmark_layernorm(rows, cols, impl)

                results[config_key][impl] = {
                    'time_ms': avg_time,
                    'bandwidth': bandwidth,
                    'name': impl_name
                }

                print(f"{impl_name:30s}: {avg_time:8.4f} ms, {bandwidth:8.2f} GB/s")
            except Exception as e:
                print(f"{impl_name:30s}: 错误 - {str(e)}")
                results[config_key][impl] = {
                    'time_ms': float('inf'),
                    'bandwidth': 0.0,
                    'name': impl_name
                }

    # 打印汇总表
    print("\n" + "=" * 80)
    print("性能汇总 (相对于 Naive 的加速比)")
    print("=" * 80)

    for rows, cols, desc in configs:
        config_key = f"{rows}x{cols}"
        if config_key not in results:
            continue

        print(f"\n{desc}:")
        print(f"{'实现':<30s} {'时间 (ms)':>12s} {'带宽 (GB/s)':>15s} {'加速比':>10s}")
        print("─" * 80)

        naive_time = results[config_key].get('naive', {}).get('time_ms', float('inf'))

        for impl, impl_name in implementations:
            if impl in results[config_key]:
                data = results[config_key][impl]
                speedup = naive_time / data['time_ms'] if naive_time > 0 else 0.0

                print(f"{data['name']:<30s} "
                      f"{data['time_ms']:>12.4f} "
                      f"{data['bandwidth']:>15.2f} "
                      f"{speedup:>10.2f}x")

    # 找出最快的实现
    print("\n" + "=" * 80)
    print("推荐实现 (按矩阵大小)")
    print("=" * 80)

    for rows, cols, desc in configs:
        config_key = f"{rows}x{cols}"
        if config_key not in results:
            continue

        # 找出最快的实现（排除 inf）
        valid_impls = [(impl, data) for impl, data in results[config_key].items()
                      if data['time_ms'] < float('inf')]

        if valid_impls:
            best_impl, best_data = min(valid_impls, key=lambda x: x[1]['time_ms'])
            print(f"{desc:30s}: {best_data['name']} ({best_data['time_ms']:.4f} ms, {best_data['bandwidth']:.2f} GB/s)")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
