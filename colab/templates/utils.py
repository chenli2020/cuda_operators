"""
Colab环境下的通用工具函数

提供测试、验证、数据生成等通用功能
"""

import torch
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json


def check_allclose(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    name: str = "",
    verbose: bool = True,
) -> bool:
    """
    检查两个张量是否接近，并打印诊断信息

    Args:
        actual: 实际输出
        expected: 期望输出
        rtol: 相对误差容忍度
        atol: 绝对误差容忍度
        name: 测试名称
        verbose: 是否打印详细信息

    Returns:
        是否通过测试
    """
    actual = actual.cpu().numpy().flatten()
    expected = expected.cpu().numpy().flatten()

    diff = np.abs(actual - expected)
    max_diff = np.max(diff)
    max_idx = np.argmax(diff)
    mean_diff = np.mean(diff)

    rel_diff = diff / (np.abs(expected) + 1e-8)
    max_rel_diff = np.max(rel_diff)

    passed = np.allclose(actual, expected, rtol=rtol, atol=atol)

    if verbose:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        print(f"    Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        print(f"    Max relative diff: {max_rel_diff:.2e}")
        if not passed:
            print(f"    At index {max_idx}: actual={actual[max_idx]:.6f}, expected={expected[max_idx]:.6f}")

    return passed


def test_operator(
    cuda_op_fn: Callable,
    torch_op_fn: Callable,
    test_cases: List[Tuple],
    rtol: float = 1e-4,
    atol: float = 1e-5,
    op_name: str = "Operator",
) -> Dict:
    """
    通用算子精度测试

    Args:
        cuda_op_fn: CUDA算子函数
        torch_op_fn: PyTorch算子函数
        test_cases: 测试用例列表
        rtol: 相对误差容忍度
        atol: 绝对误差容忍度
        op_name: 算子名称

    Returns:
        测试结果字典
    """
    print("\n" + "=" * 80)
    print(f"{op_name} Accuracy Test")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return {}

    device = torch.device('cuda')
    all_passed = True
    results = {}

    for i, config in enumerate(test_cases):
        if isinstance(config, int):
            size = config
            print(f"\nTest case {i+1}: size={size}")
        else:
            rows, cols = config
            print(f"\nTest case {i+1}: rows={rows}, cols={cols}")

        results[f"case_{i+1}"] = {}

        # 创建测试数据
        torch.manual_seed(42)
        if isinstance(config, int):
            input_t = torch.randn(config, device=device, dtype=torch.float32)
        else:
            # 确保对齐到4（用于向量化版本）
            if isinstance(config, tuple) and len(config) == 2:
                rows, cols = config
                cols_aligned = (cols // 4) * 4
                if cols_aligned == 0:
                    cols_aligned = 4
                input_t = torch.randn(rows, cols_aligned, device=device, dtype=torch.float32)
            else:
                input_t = torch.randn(*config, device=device, dtype=torch.float32)

        try:
            # PyTorch参考实现
            expected = torch_op_fn(input_t)

            # CUDA实现
            actual = cuda_op_fn(input_t)

            # 检查精度
            passed = check_allclose(actual, expected, rtol=rtol, atol=atol, name=f"  {op_name}")
            all_passed = all_passed and passed

            results[f"case_{i+1}"]['passed'] = passed
            results[f"case_{i+1}"]['max_diff'] = float(torch.max(torch.abs(actual - expected)))
            results[f"case_{i+1}"]['mean_diff'] = float(torch.mean(torch.abs(actual - expected)))

        except Exception as e:
            print(f"  Error: {e}")
            results[f"case_{i+1}"]['error'] = str(e)
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All tests PASSED!")
    else:
        print("✗ Some tests FAILED!")

    return {'all_passed': all_passed, 'cases': results}


def generate_test_data(
    rows: int,
    cols: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """
    生成测试数据

    Args:
        rows: 行数
        cols: 列数
        device: 设备
        dtype: 数据类型
        seed: 随机种子

    Returns:
        测试数据张量
    """
    torch.manual_seed(seed)
    return torch.randn(rows, cols, device=device, dtype=dtype)


def plot_performance_comparison(
    configs: List[Tuple],
    times_dict: Dict[str, List[float]],
    title: str = "Performance Comparison",
    ylabel: str = "Time (ms)",
    xlabel: str = "Configuration",
    log_scale: bool = True,
) -> None:
    """
    绘制性能对比图

    Args:
        configs: 测试配置列表
        times_dict: 时间字典 {impl_name: [times]}
        title: 图表标题
        ylabel: Y轴标签
        xlabel: X轴标签
        log_scale: 是否使用对数刻度
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x_labels = [str(c) for c in configs]
    x = np.arange(len(x_labels))

    for impl_name, times in times_dict.items():
        ax.plot(x, times, marker='o', label=impl_name, linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.show()


def plot_speedup_bars(
    configs: List[Tuple],
    speedup_dict: Dict[str, List[float]],
    baseline: str = "PyTorch",
    title: str = "Speedup Comparison",
) -> None:
    """
    绘制加速比柱状图

    Args:
        configs: 测试配置列表
        speedup_dict: 加速比字典 {impl_name: [speedups]}
        baseline: 基线实现名称
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x_labels = [str(c) for c in configs]
    x = np.arange(len(x_labels))

    width = 0.8 / len(speedup_dict)
    offsets = np.linspace(-width * len(speedup_dict) / 2, width * len(speedup_dict) / 2, len(speedup_dict))

    for (impl_name, speedups), offset in zip(speedup_dict.items(), offsets):
        ax.bar(x + offset, speedups, width, label=impl_name, alpha=0.8)

    ax.set_xlabel("Configuration")
    ax.set_ylabel(f"Speedup vs {baseline}")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1, color='r', linestyle='--', label=f'{baseline} baseline')

    plt.tight_layout()
    plt.show()


def save_benchmark_results(results: Dict, filename: str) -> None:
    """
    保存基准测试结果到JSON文件

    Args:
        results: 测试结果字典
        filename: 文件名
    """
    # 转换numpy类型为Python原生类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results_converted = convert(results)

    with open(filename, 'w') as f:
        json.dump(results_converted, f, indent=2)

    print(f"✓ Results saved to {filename}")


def load_benchmark_results(filename: str) -> Dict:
    """
    从JSON文件加载基准测试结果

    Args:
        filename: 文件名

    Returns:
        测试结果字典
    """
    with open(filename, 'r') as f:
        results = json.load(f)

    print(f"✓ Results loaded from {filename}")
    return results


def print_test_summary(
    test_results: Dict,
    op_name: str = "Operator",
) -> None:
    """
    打印测试总结

    Args:
        test_results: 测试结果字典
        op_name: 算子名称
    """
    print("\n" + "=" * 80)
    print(f"{op_name} Test Summary")
    print("=" * 80)

    total_cases = len([k for k in test_results.keys() if k.startswith('case_')])
    passed_cases = sum(1 for k, v in test_results.items() if k.startswith('case_') and v.get('passed', False))

    print(f"\nTotal Test Cases: {total_cases}")
    print(f"Passed: {passed_cases}")
    print(f"Failed: {total_cases - passed_cases}")

    if passed_cases == total_cases:
        print("\n✓ All tests PASSED!")
    else:
        print("\n✗ Some tests FAILED!")
        for k, v in test_results.items():
            if k.startswith('case_') and not v.get('passed', False):
                print(f"  - {k}: {v.get('error', 'Failed')}")


def create_comparison_table(
    configs: List[Tuple],
    metrics_dict: Dict[str, List[Dict]],
    metrics_to_show: List[str] = ['time_ms', 'bandwidth_gb_s', 'bandwidth_utilization'],
) -> None:
    """
    创建性能对比表格

    Args:
        configs: 测试配置列表
        metrics_dict: 指标字典 {impl_name: [metrics]}
        metrics_to_show: 要显示的指标列表
    """
    print("\n" + "=" * 100)
    print(f"{'Configuration':>15}", end='')

    for impl_name in metrics_dict.keys():
        print(f" | {impl_name:>20}", end='')
    print("\n" + "=" * 100)

    for i, config in enumerate(configs):
        print(f"{str(config):>15}", end='')

        for impl_name, metrics_list in metrics_dict.items():
            if i < len(metrics_list):
                metrics = metrics_list[i]
                if metrics is None:
                    print(f" | {'Error':>20}", end='')
                else:
                    # 显示时间
                    time_str = f"{metrics['time_ms']:.3f}ms"
                    print(f" | {time_str:>20}", end='')
            else:
                print(f" | {'N/A':>20}", end='')

        print()

    print("\nDetailed Metrics:")
    for i, config in enumerate(configs):
        print(f"\nConfiguration: {config}")
        for impl_name, metrics_list in metrics_dict.items():
            if i < len(metrics_list) and metrics_list[i] is not None:
                print(f"  {impl_name}:")
                metrics = metrics_list[i]
                for metric_name in metrics_to_show:
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        if 'utilization' in metric_name or 'percent' in metric_name:
                            print(f"    {metric_name}: {value:.1f}%")
                        elif 'time' in metric_name:
                            print(f"    {metric_name}: {value:.3f} ms")
                        else:
                            print(f"    {metric_name}: {value:.2f}")


if __name__ == "__main__":
    # 测试代码
    print("Colab Utility Functions")
    print("=" * 50)

    # 测试check_allclose
    x = torch.randn(10, 10)
    y = x + 1e-6
    check_allclose(x, y, rtol=1e-4, atol=1e-5, name="Test")
