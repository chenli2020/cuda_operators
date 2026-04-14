"""Tests for layernorm operations."""
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


def pytorch_layernorm(x, weight, bias, eps=1e-5):
    """PyTorch layernorm reference."""
    t = torch.from_numpy(x)
    w = torch.from_numpy(weight)
    b = torch.from_numpy(bias)
    ln = torch.nn.LayerNorm(x.shape[-1], elementwise_affine=True, eps=eps)
    ln.weight.data = w
    ln.bias.data = b
    with torch.no_grad():
        return ln(t).numpy()


def test_layernorm():
    """Test layernorm operation."""
    print("Testing layernorm...")

    test_cases = [
        (2, 128),
        (32, 512),
        (32, 1024),
        (128, 4096),
        (128, 8192),
    ]

    all_passed = True
    for rows, cols in test_cases:
        input_np = generate_random_input((rows, cols))
        weight_np = np.ones(cols, dtype=np.float32)
        bias_np = np.zeros(cols, dtype=np.float32)

        expected = pytorch_layernorm(input_np, weight_np, bias_np)

        for impl in ["naive", "warp", "vectorized", "auto"]:
            try:
                actual = cuda_ops.layernorm(input_np, weight_np, bias_np,
                                            rows, cols, 1e-5, impl)
                passed = check_allclose(actual, expected, rtol=1e-4, atol=1e-5,
                                        name=f"rows={rows}, cols={cols}, impl={impl}")
                all_passed = all_passed and passed
            except Exception as e:
                print(f"  Error with {impl}: {e}")
                all_passed = False

    return all_passed


def benchmark_layernorm():
    """Benchmark layernorm operations."""
    print("\nBenchmarking layernorm...")

    import time

    test_cases = [
        (32, 512),
        (32, 1024),
        (128, 4096),
        (256, 8192),
    ]

    for rows, cols in test_cases:
        input_np = generate_random_input((rows, cols))
        weight_np = np.ones(cols, dtype=np.float32)
        bias_np = np.zeros(cols, dtype=np.float32)

        input_torch = torch.from_numpy(input_np).cuda()
        weight_torch = torch.from_numpy(weight_np).cuda()
        bias_torch = torch.from_numpy(bias_np).cuda()

        ln = torch.nn.LayerNorm(cols, elementwise_affine=True, eps=1e-5).cuda()
        ln.weight.data = weight_torch
        ln.bias.data = bias_torch

        # Warmup
        for _ in range(10):
            _ = cuda_ops.layernorm(input_np, weight_np, bias_np, rows, cols, 1e-5, "auto")
            _ = ln(input_torch)

        torch.cuda.synchronize()

        # Benchmark CUDA
        num_iters = 100
        start = time.time()
        for _ in range(num_iters):
            _ = cuda_ops.layernorm(input_np, weight_np, bias_np, rows, cols, 1e-5, "auto")
        cuda_time = (time.time() - start) / num_iters * 1000

        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = ln(input_torch)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / num_iters * 1000

        # Compute bandwidth
        bytes_per_elem = 4
        # Read: input, weight, bias; Write: output
        # Plus intermediate computations
        total_bytes = (rows * cols * 3 + cols * 2) * bytes_per_elem
        cuda_bw = total_bytes / (cuda_time / 1000) / 1e9
        torch_bw = total_bytes / (torch_time / 1000) / 1e9

        print(f"  [{rows:4}, {cols:4}]: CUDA={cuda_time:.3f}ms ({cuda_bw:.2f}GB/s), "
              f"PyTorch={torch_time:.3f}ms ({torch_bw:.2f}GB/s)")


if __name__ == "__main__":
    test_layernorm()
    benchmark_layernorm()
