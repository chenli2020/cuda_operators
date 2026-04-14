"""Tests for softmax operations."""
import numpy as np
import torch
import sys
import os

try:
    import cuda_ops
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
    import cuda_ops

from utils import check_allclose, generate_random_input


def pytorch_softmax(x):
    """PyTorch softmax reference."""
    t = torch.from_numpy(x)
    return torch.nn.functional.softmax(t, dim=-1).numpy()


def test_softmax():
    """Test softmax operation."""
    print("Testing softmax...")

    test_cases = [
        (2, 128),
        (32, 128),
        (32, 512),
        (32, 1024),
        (128, 4096),
    ]

    all_passed = True
    for rows, cols in test_cases:
        input_np = generate_random_input((rows, cols))
        expected = pytorch_softmax(input_np)

        for impl in ["naive", "online", "warp", "auto"]:
            try:
                actual = cuda_ops.softmax(input_np, rows, cols, impl)
                passed = check_allclose(actual, expected, rtol=1e-4, atol=1e-5,
                                        name=f"rows={rows}, cols={cols}, impl={impl}")
                all_passed = all_passed and passed
            except Exception as e:
                print(f"  Error with {impl}: {e}")
                all_passed = False

    return all_passed


def benchmark_softmax():
    """Benchmark softmax operations."""
    print("\nBenchmarking softmax...")

    import time

    test_cases = [
        (32, 512),
        (32, 1024),
        (128, 4096),
        (256, 8192),
    ]

    for rows, cols in test_cases:
        input_np = generate_random_input((rows, cols))
        input_torch = torch.from_numpy(input_np).cuda()

        # Warmup
        for _ in range(10):
            _ = cuda_ops.softmax(input_np, rows, cols, "auto")
            _ = torch.nn.functional.softmax(input_torch, dim=-1)

        torch.cuda.synchronize()

        # Benchmark CUDA
        num_iters = 100
        start = time.time()
        for _ in range(num_iters):
            _ = cuda_ops.softmax(input_np, rows, cols, "auto")
        cuda_time = (time.time() - start) / num_iters * 1000

        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = torch.nn.functional.softmax(input_torch, dim=-1)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / num_iters * 1000

        # Compute throughput
        elements = rows * cols
        cuda_throughput = elements / (cuda_time / 1000) / 1e6  # M elems/s
        torch_throughput = elements / (torch_time / 1000) / 1e6

        print(f"  [{rows:4}, {cols:4}]: CUDA={cuda_time:.3f}ms ({cuda_throughput:.1f}M/s), "
              f"PyTorch={torch_time:.3f}ms ({torch_throughput:.1f}M/s)")


if __name__ == "__main__":
    test_softmax()
    benchmark_softmax()
