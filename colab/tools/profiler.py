"""
Colab环境下的CUDA性能分析工具

由于Colab不支持Nsight，使用PyTorch内置profiler替代
提供统一的性能分析接口和带宽/计算利用率计算
"""

import torch
import time
from typing import Callable, Dict, List, Tuple, Optional
from contextlib import contextmanager


class PerformanceMetrics:
    """性能指标数据类"""

    def __init__(self):
        self.time_ms = 0.0
        self.bandwidth_gb_s = 0.0
        self.bandwidth_utilization = 0.0  # 百分比
        self.throughput_elems_s = 0.0
        self.gflops = 0.0
        self.compute_utilization = 0.0  # 百分比

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'time_ms': self.time_ms,
            'bandwidth_gb_s': self.bandwidth_gb_s,
            'bandwidth_utilization': self.bandwidth_utilization,
            'throughput_elems_s': self.throughput_elems_s,
            'gflops': self.gflops,
            'compute_utilization': self.compute_utilization,
        }

    def __str__(self) -> str:
        """格式化输出"""
        lines = [
            f"Time: {self.time_ms:.3f} ms",
        ]
        if self.bandwidth_gb_s > 0:
            lines.extend([
                f"Bandwidth: {self.bandwidth_gb_s:.2f} GB/s",
                f"Bandwidth Utilization: {self.bandwidth_utilization:.1f}%",
            ])
        if self.gflops > 0:
            lines.extend([
                f"GFLOPS: {self.gflops:.2f}",
                f"Compute Utilization: {self.compute_utilization:.1f}%",
            ])
        if self.throughput_elems_s > 0:
            lines.append(f"Throughput: {self.throughput_elems_s/1e6:.2f} M elems/s")
        return "\n".join(lines)


def get_gpu_info() -> Dict:
    """获取GPU信息"""
    if not torch.cuda.is_available():
        return {'available': False}

    device_name = torch.cuda.get_device_name(0)
    device_capability = torch.cuda.get_device_capability(0)
    device_properties = torch.cuda.get_device_properties(0)

    # 根据GPU类型设置理论峰值
    gpu_info = {
        'available': True,
        'name': device_name,
        'capability': device_capability,
        'sm_count': device_properties.multi_processor_count,
        'total_memory_gb': device_properties.total_memory / 1e9,
    }

    # 根据GPU型号设置理论带宽和计算峰值
    # T4: 320 GB/s带宽, 8.1 TFLOPS (FP32)
    # V100: 900 GB/s带宽, 15.7 TFLOPS (FP32)
    # A100: 1.6 TB/s带宽, 19.5 TFLOPS (FP32)
    if 'T4' in device_name or 'Tesla T4' in device_name:
        gpu_info.update({
            'peak_bandwidth_gb_s': 320.0,
            'peak_compute_tflops': 8.1,
        })
    elif 'V100' in device_name:
        gpu_info.update({
            'peak_bandwidth_gb_s': 900.0,
            'peak_compute_tflops': 15.7,
        })
    elif 'A100' in device_name:
        gpu_info.update({
            'peak_bandwidth_gb_s': 1600.0,
            'peak_compute_tflops': 19.5,
        })
    else:
        # 默认值（保守估计）
        gpu_info.update({
            'peak_bandwidth_gb_s': 500.0,
            'peak_compute_tflops': 10.0,
        })

    return gpu_info


@contextmanager
def profile_cuda():
    """
    使用PyTorch profiler分析CUDA kernel

    使用示例:
        with profile_cuda() as prof:
            output = cuda_ops.layernorm(input, weight, bias)
        print(prof.get_summary())
    """
    if not torch.cuda.is_available():
        yield None
        return

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as p:
        yield ProfilerWrapper(p)


class ProfilerWrapper:
    """Profiler包装器，提供便捷的查询接口"""

    def __init__(self, profiler: torch.profiler.profile):
        self.profiler = profiler

    def get_summary(self, sort_by: str = "cuda_time_total") -> str:
        """获取profiler总结"""
        return self.profiler.key_averages().table(sort_by=sort_by)

    def get_kernel_stats(self) -> Dict:
        """获取kernel级别的统计信息"""
        events = self.profiler.key_averages()
        stats = {
            'total_cuda_time_ms': 0.0,
            'kernels': [],
        }

        for event in events:
            if event.device_type == torch.profiler.DeviceType.CUDA:
                stats['total_cuda_time_ms'] += event.cuda_time_total
                stats['kernels'].append({
                    'name': event.key,
                    'cuda_time_ms': event.cuda_time_total,
                    'cuda_time_percent': event.cuda_time_total / self.profiler.profiler().metadata()[0] if self.profiler.profiler().metadata() else 0,
                })

        return stats

    def get_memory_stats(self) -> Dict:
        """获取内存统计信息"""
        # 这里简化处理，实际可以从profiler中提取更详细的信息
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1e6,
            'reserved_mb': torch.cuda.memory_reserved() / 1e6,
        }


def benchmark_function(
    func: Callable,
    *args,
    num_warmup: int = 10,
    num_iters: int = 100,
    **kwargs
) -> Tuple[any, PerformanceMetrics]:
    """
    性能测试函数

    Args:
        func: 要测试的函数
        *args: 函数参数
        num_warmup: 预热次数
        num_iters: 测试迭代次数
        **kwargs: 函数关键字参数

    Returns:
        (函数输出, 性能指标)
    """
    if not torch.cuda.is_available():
        result = func(*args, **kwargs)
        return result, PerformanceMetrics()

    # 预热
    for _ in range(num_warmup):
        _ = func(*args, **kwargs)
    torch.cuda.synchronize()

    # 计时
    start = time.time()
    for _ in range(num_iters):
        result = func(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / num_iters * 1000  # ms

    metrics = PerformanceMetrics()
    metrics.time_ms = elapsed

    return result, metrics


def calculate_bandwidth_utilization(
    rows: int,
    cols: int,
    time_ms: float,
    num_reads: int = 3,  # input + weight + bias
    num_writes: int = 1,  # output
) -> Tuple[float, float]:
    """
    计算内存带宽利用率

    Args:
        rows: 行数
        cols: 列数
        time_ms: 执行时间（毫秒）
        num_reads: 读取次数
        num_writes: 写入次数

    Returns:
        (带宽GB/s, 利用率百分比)
    """
    gpu_info = get_gpu_info()
    if not gpu_info['available']:
        return 0.0, 0.0

    # 计算数据量（字节）
    bytes_per_elem = 4  # float32
    total_bytes = (rows * cols * num_reads + rows * cols * num_writes + cols * 2) * bytes_per_elem

    # 计算带宽
    bandwidth_gb_s = total_bytes / (time_ms / 1000) / 1e9

    # 计算利用率
    peak_bandwidth = gpu_info['peak_bandwidth_gb_s']
    utilization = (bandwidth_gb_s / peak_bandwidth) * 100

    return bandwidth_gb_s, utilization


def calculate_compute_utilization(
    m: int,
    n: int,
    k: int,
    time_ms: float,
) -> Tuple[float, float]:
    """
    计算计算利用率（用于MatMul等计算密集型算子）

    Args:
        m, n, k: 矩阵维度
        time_ms: 执行时间（毫秒）

    Returns:
        (GFLOPS, 利用率百分比)
    """
    gpu_info = get_gpu_info()
    if not gpu_info['available']:
        return 0.0, 0.0

    # 计算FLOPS（矩阵乘法：2*m*n*k）
    flops = 2 * m * n * k
    gflops = flops / (time_ms / 1000) / 1e9

    # 计算利用率
    peak_tflops = gpu_info['peak_compute_tflops']
    peak_gflops = peak_tflops * 1000
    utilization = (gflops / peak_gflops) * 100

    return gflops, utilization


def print_gpu_info():
    """打印GPU信息"""
    gpu_info = get_gpu_info()

    if not gpu_info['available']:
        print("❌ CUDA is not available")
        return

    print("✅ CUDA GPU Information:")
    print(f"  GPU: {gpu_info['name']}")
    print(f"  Compute Capability: {gpu_info['capability'][0]}.{gpu_info['capability'][1]}")
    print(f"  SM Count: {gpu_info['sm_count']}")
    print(f"  Total Memory: {gpu_info['total_memory_gb']:.2f} GB")
    print(f"  Peak Bandwidth: {gpu_info['peak_bandwidth_gb_s']:.0f} GB/s")
    print(f"  Peak Compute: {gpu_info['peak_compute_tflops']:.1f} TFLOPS")


def benchmark_operator(
    cuda_op: Callable,
    torch_op: Callable,
    test_configs: List[Tuple],
    op_name: str = "Operator",
) -> Dict:
    """
    算子性能基准测试

    Args:
        cuda_op: CUDA算子函数
        torch_op: PyTorch算子函数
        test_configs: 测试配置列表 [(rows, cols), ...]
        op_name: 算子名称

    Returns:
        测试结果字典
    """
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return {}

    gpu_info = get_gpu_info()
    print(f"\n{'='*80}")
    print(f"{op_name} Performance Benchmark")
    print(f"GPU: {gpu_info['name']}")
    print(f"{'='*80}")

    results = {}

    for config in test_configs:
        if isinstance(config, int):
            # Reduce等一维算子
            size = config
            shape_str = f"[{size}]"
            print(f"\nConfiguration: {shape_str}")
        else:
            # 二维算子
            rows, cols = config
            shape_str = f"[{rows}, {cols}]"
            print(f"\nConfiguration: {shape_str}")

        results[shape_str] = {}

        # 创建测试数据
        if isinstance(config, int):
            data = torch.randn(config, device='cuda', dtype=torch.float32)
        else:
            data = torch.randn(rows, cols, device='cuda', dtype=torch.float32)

        # 测试CUDA实现
        try:
            _, cuda_metrics = benchmark_function(cuda_op, data)
            results[shape_str]['cuda'] = cuda_metrics

            # 计算带宽利用率
            if isinstance(config, int):
                bw, util = calculate_bandwidth_utilization(1, config, cuda_metrics.time_ms, num_reads=1, num_writes=1)
            else:
                bw, util = calculate_bandwidth_utilization(rows, cols, cuda_metrics.time_ms)
            cuda_metrics.bandwidth_gb_s = bw
            cuda_metrics.bandwidth_utilization = util

            print(f"  CUDA: {cuda_metrics.time_ms:.3f} ms | {bw:.2f} GB/s ({util:.1f}%)")
        except Exception as e:
            print(f"  CUDA: Error - {e}")
            results[shape_str]['cuda'] = None

        # 测试PyTorch实现
        try:
            _, torch_metrics = benchmark_function(torch_op, data)
            results[shape_str]['torch'] = torch_metrics

            # 计算带宽利用率
            if isinstance(config, int):
                bw, util = calculate_bandwidth_utilization(1, config, torch_metrics.time_ms, num_reads=1, num_writes=1)
            else:
                bw, util = calculate_bandwidth_utilization(rows, cols, torch_metrics.time_ms)
            torch_metrics.bandwidth_gb_s = bw
            torch_metrics.bandwidth_utilization = util

            print(f"  PyTorch: {torch_metrics.time_ms:.3f} ms | {bw:.2f} GB/s ({util:.1f}%)")

            # 计算加速比
            if results[shape_str]['cuda'] is not None:
                speedup = torch_metrics.time_ms / cuda_metrics.time_ms
                print(f"  Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"  PyTorch: Error - {e}")
            results[shape_str]['torch'] = None

    return results


if __name__ == "__main__":
    # 测试代码
    print("Colab Performance Analysis Tools")
    print("=" * 50)

    print_gpu_info()

    # 简单的性能测试
    if torch.cuda.is_available():
        def test_func(x):
            return x * 2

        data = torch.randn(1000, 1000, device='cuda')
        result, metrics = benchmark_function(test_func, data)
        print(f"\nTest function execution:")
        print(f"  {metrics}")
