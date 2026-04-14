"""Tests for matmul operations."""
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


def pytorch_matmul(A, B):
    """PyTorch matmul reference."""
    t_A = torch.from_numpy(A)
    t_B = torch.from_numpy(B)
    with torch.no_grad():
        return torch.matmul(t_A, t_B).numpy()


def test_matmul():
    """Test matmul operation."""
    print("Testing matmul...")

    test_cases = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]

    all_passed = True
    for M, N, K in test_cases:
        A_np = generate_random_input((M, K))
        B_np = generate_random_input((K, N))

        expected = pytorch_matmul(A_np, B_np)

        for impl in ["naive", "shared", "2d_tiling", "auto"]:
            try:
                actual = cuda_ops.matmul(A_np, B_np, M, N, K, impl)
                passed = check_allclose(actual, expected, rtol=1e-3, atol=1e-4,
                                        name=f"M={M}, N={N}, K={K}, impl={impl}")
                all_passed = all_passed and passed
            except Exception as e:
                print(f"  Error with {impl}: {e}")
                all_passed = False

    return all_passed


def benchmark_matmul():
    """Benchmark matmul operations."""
    print("\nBenchmarking matmul...")

    import time

    test_cases = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    for M, N, K in test_cases:
        A_np = generate_random_input((M, K))
        B_np = generate_random_input((K, N))

        A_torch = torch.from_numpy(A_np).cuda()
        B_torch = torch.from_numpy(B_np).cuda()

        # Warmup
        for _ in range(10):
            _ = cuda_ops.matmul(A_np, B_np, M, N, K, "auto")
            _ = torch.matmul(A_torch, B_torch)

        torch.cuda.synchronize()

        # Benchmark CUDA
        num_iters = 50
        start = time.time()
        for _ in range(num_iters):
            _ = cuda_ops.matmul(A_np, B_np, M, N, K, "auto")
        cuda_time = (time.time() - start) / num_iters * 1000

        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = torch.matmul(A_torch, B_torch)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / num_iters * 1000

        # Compute GFLOPS
        flops = 2 * M * N * K
        cuda_gflops = flops / (cuda_time / 1000) / 1e9
        torch_gflops = flops / (torch_time / 1000) / 1e9

        print(f"  [{M:4}, {N:4}, {K:4}]: CUDA={cuda_time:.3f}ms ({cuda_gflops:.1f}GFLOPS), "
              f"PyTorch={torch_time:.3f}ms ({torch_gflops:.1f}GFLOPS)")


if __name__ == "__main__":
    test_matmul()
    benchmark_matmul()
