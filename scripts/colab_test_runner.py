#!/usr/bin/env python3
"""
Colab测试辅助脚本

自动检测Colab环境，批量运行所有算子的测试，生成统一报告。
"""

import os
import sys
import time
import json
from typing import Dict, List, Optional
import subprocess


def is_colab() -> bool:
    """检测是否在Colab环境中运行"""
    return 'google.colab' in sys.modules


def check_gpu_available() -> bool:
    """检查GPU是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_info() -> Dict:
    """获取GPU信息"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {'available': False}

        return {
            'available': True,
            'name': torch.cuda.get_device_name(0),
            'capability': torch.cuda.get_device_capability(0),
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    except ImportError:
        return {'available': False}


def run_notebook(notebook_path: str) -> Dict:
    """
    运行Jupyter notebook并返回结果

    Args:
        notebook_path: notebook文件路径

    Returns:
        运行结果字典
    """
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor

        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # 配置执行处理器
        ep = ExecutePreprocessor(
            timeout=600,  # 10分钟超时
            kernel_name='python3',
        )

        # 执行notebook
        start_time = time.time()
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        elapsed = time.time() - start_time

        return {
            'success': True,
            'time': elapsed,
            'error': None,
        }

    except Exception as e:
        return {
            'success': False,
            'time': 0,
            'error': str(e),
        }


def test_operator(
    operator_name: str,
    notebook_path: str,
) -> Dict:
    """
    测试单个算子

    Args:
        operator_name: 算子名称
        notebook_path: notebook路径

    Returns:
        测试结果
    """
    print(f"\n{'='*60}")
    print(f"Testing {operator_name}")
    print(f"{'='*60}")

    if not os.path.exists(notebook_path):
        return {
            'operator': operator_name,
            'success': False,
            'error': f'Notebook not found: {notebook_path}',
        }

    print(f"Running {notebook_path}...")

    result = run_notebook(notebook_path)
    result['operator'] = operator_name

    if result['success']:
        print(f"✓ {operator_name} PASSED ({result['time']:.1f}s)")
    else:
        print(f"✗ {operator_name} FAILED")
        print(f"  Error: {result['error']}")

    return result


def run_all_tests(
    operators: Optional[List[str]] = None,
) -> Dict:
    """
    运行所有算子的测试

    Args:
        operators: 要测试的算子列表，None表示测试所有

    Returns:
        测试报告
    """
    # 定义所有可用的算子
    all_operators = {
        'layernorm': 'colab_layernorm.ipynb',
        'rmsnorm': 'colab_rmsnorm.ipynb',
        'softmax': 'colab_softmax.ipynb',
        'reduce': 'colab_reduce.ipynb',
        'matmul': 'colab_matmul.ipynb',
    }

    # 确定要测试的算子
    if operators is None:
        operators = list(all_operators.keys())
    else:
        # 验证算子名称
        for op in operators:
            if op not in all_operators:
                print(f"Warning: Unknown operator '{op}', skipping")

    # 获取GPU信息
    gpu_info = get_gpu_info()

    # 运行测试
    results = {
        'environment': {
            'is_colab': is_colab(),
            'gpu': gpu_info,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'results': [],
    }

    for op_name in operators:
        if op_name in all_operators:
            notebook_path = all_operators[op_name]
            result = test_operator(op_name, notebook_path)
            results['results'].append(result)

    # 统计
    total = len(results['results'])
    passed = sum(1 for r in results['results'] if r['success'])
    failed = total - passed

    results['summary'] = {
        'total': total,
        'passed': passed,
        'failed': failed,
    }

    return results


def print_report(results: Dict) -> None:
    """打印测试报告"""
    print("\n" + "="*80)
    print("TEST REPORT")
    print("="*80)

    # 环境信息
    print("\nEnvironment:")
    print(f"  Platform: {'Colab' if results['environment']['is_colab'] else 'Local'}")
    if results['environment']['gpu']['available']:
        gpu = results['environment']['gpu']
        print(f"  GPU: {gpu['name']}")
        print(f"  Memory: {gpu['memory_gb']:.2f} GB")
    else:
        print(f"  GPU: Not available")

    # 测试结果
    print("\nResults:")
    for result in results['results']:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"  {result['operator']:12} {status}")
        if not result['success'] and result.get('error'):
            print(f"    Error: {result['error']}")

    # 总结
    summary = results['summary']
    print("\nSummary:")
    print(f"  Total: {summary['total']}")
    print(f"  Passed: {summary['passed']}")
    print(f"  Failed: {summary['failed']}")

    if summary['failed'] == 0:
        print("\n✓ All tests PASSED!")
    else:
        print(f"\n✗ {summary['failed']} test(s) FAILED")


def save_report(results: Dict, filename: str = "test_report.json") -> None:
    """保存测试报告到文件"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Report saved to {filename}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Colab Test Runner')
    parser.add_argument(
        '--ops',
        nargs='+',
        choices=['layernorm', 'rmsnorm', 'softmax', 'reduce', 'matmul'],
        help='Operators to test (default: all)'
    )
    parser.add_argument(
        '--output',
        default='test_report.json',
        help='Output report filename'
    )
    parser.add_argument(
        '--skip-run',
        action='store_true',
        help='Skip running tests, only print environment info'
    )

    args = parser.parse_args()

    # 打印环境信息
    print("="*80)
    print("CUDA Operators Test Runner")
    print("="*80)

    gpu_info = get_gpu_info()
    print(f"\nEnvironment:")
    print(f"  Platform: {'Colab' if is_colab() else 'Local'}")
    if gpu_info['available']:
        print(f"  GPU: {gpu_info['name']}")
        print(f"  Compute Capability: {gpu_info['capability'][0]}.{gpu_info['capability'][1]}")
        print(f"  Memory: {gpu_info['memory_gb']:.2f} GB")
    else:
        print(f"  GPU: Not available")
        print(f"\n✗ Error: GPU is required for CUDA tests")
        sys.exit(1)

    if args.skip_run:
        print("\nSkipping tests (--skip-run flag set)")
        return

    # 运行测试
    operators = args.ops
    results = run_all_tests(operators)

    # 打印报告
    print_report(results)

    # 保存报告
    save_report(results, args.output)

    # 返回退出码
    sys.exit(0 if results['summary']['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
