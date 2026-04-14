"""Comprehensive benchmark suite for CUDA operators."""
import numpy as np
import torch
import sys
import os
import json
import time
from datetime import datetime

try:
    import cuda_ops
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
    import cuda_ops


class BenchmarkRunner:
    """Run benchmarks and collect results."""

    def __init__(self, num_warmup=10, num_iterations=100):
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.results = {}

    def benchmark_reduce(self):
        """Benchmark reduce operations."""
        print("\n" + "="*60)
        print("Reduce Sum Benchmark")
        print("="*60)

        sizes = [10000, 100000, 1000000, 10000000]
        results = []

        for n in sizes:
            input_np = np.random.randn(n).astype(np.float32)

            # Warmup
            for _ in range(self.num_warmup):
                _ = cuda_ops.reduce_sum(input_np, "auto")

            # Benchmark
            times = []
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                _ = cuda_ops.reduce_sum(input_np, "auto")
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg_time = np.mean(times)
            min_time = np.min(times)
            # Bandwidth: read n floats
            bandwidth_gb_s = (n * 4) / (avg_time / 1000) / 1e9

            results.append({
                'size': n,
                'avg_ms': avg_time,
                'min_ms': min_time,
                'bandwidth_gb_s': bandwidth_gb_s
            })

            print(f"  Size {n:>10}: {avg_time:.3f}ms (min: {min_time:.3f}ms), "
                  f"BW: {bandwidth_gb_s:.2f} GB/s")

        self.results['reduce'] = results
        return results

    def benchmark_softmax(self):
        """Benchmark softmax operations."""
        print("\n" + "="*60)
        print("Softmax Benchmark")
        print("="*60)

        test_cases = [
            (32, 512),
            (32, 1024),
            (128, 4096),
            (256, 8192),
        ]
        results = []

        for rows, cols in test_cases:
            input_np = np.random.randn(rows, cols).astype(np.float32)

            # Warmup
            for _ in range(self.num_warmup):
                _ = cuda_ops.softmax(input_np, rows, cols, "auto")

            # Benchmark
            times = []
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                _ = cuda_ops.softmax(input_np, rows, cols, "auto")
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg_time = np.mean(times)
            throughput = (rows * cols) / (avg_time / 1000) / 1e6

            results.append({
                'shape': [rows, cols],
                'avg_ms': avg_time,
                'throughput_melems_s': throughput
            })

            print(f"  [{rows:4}, {cols:4}]: {avg_time:.3f}ms, "
                  f"{throughput:.1f}M elems/s")

        self.results['softmax'] = results
        return results

    def benchmark_norm(self):
        """Benchmark LayerNorm and RMSNorm."""
        print("\n" + "="*60)
        print("Normalization Benchmark")
        print("="*60)

        test_cases = [
            (32, 512),
            (32, 1024),
            (128, 4096),
        ]
        results = {'layernorm': [], 'rmsnorm': []}

        for rows, cols in test_cases:
            input_np = np.random.randn(rows, cols).astype(np.float32)
            weight_np = np.ones(cols, dtype=np.float32)
            bias_np = np.zeros(cols, dtype=np.float32)

            # LayerNorm
            for _ in range(self.num_warmup):
                _ = cuda_ops.layernorm(input_np, weight_np, bias_np,
                                       rows, cols, 1e-5, "auto")

            times = []
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                _ = cuda_ops.layernorm(input_np, weight_np, bias_np,
                                       rows, cols, 1e-5, "auto")
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg_time = np.mean(times)
            results['layernorm'].append({
                'shape': [rows, cols],
                'avg_ms': avg_time
            })
            print(f"  LayerNorm [{rows:4}, {cols:4}]: {avg_time:.3f}ms")

            # RMSNorm
            for _ in range(self.num_warmup):
                _ = cuda_ops.rmsnorm(input_np, weight_np, rows, cols, 1e-5, "auto")

            times = []
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                _ = cuda_ops.rmsnorm(input_np, weight_np, rows, cols, 1e-5, "auto")
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg_time = np.mean(times)
            results['rmsnorm'].append({
                'shape': [rows, cols],
                'avg_ms': avg_time
            })
            print(f"  RMSNorm   [{rows:4}, {cols:4}]: {avg_time:.3f}ms")

        self.results['norm'] = results
        return results

    def benchmark_matmul(self):
        """Benchmark matrix multiplication."""
        print("\n" + "="*60)
        print("Matrix Multiplication Benchmark")
        print("="*60)

        test_cases = [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
        ]
        results = []

        for M, N, K in test_cases:
            A_np = np.random.randn(M, K).astype(np.float32)
            B_np = np.random.randn(K, N).astype(np.float32)

            # Warmup
            for _ in range(self.num_warmup):
                _ = cuda_ops.matmul(A_np, B_np, M, N, K, "auto")

            # Benchmark
            times = []
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                _ = cuda_ops.matmul(A_np, B_np, M, N, K, "auto")
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg_time = np.mean(times)
            flops = 2 * M * N * K
            gflops = flops / (avg_time / 1000) / 1e9

            results.append({
                'shape': [M, N, K],
                'avg_ms': avg_time,
                'gflops': gflops
            })

            print(f"  [{M:4}, {N:4}, {K:4}]: {avg_time:.3f}ms, {gflops:.1f} GFLOPS")

        self.results['matmul'] = results
        return results

    def run_all(self):
        """Run all benchmarks."""
        print("\n" + "="*60)
        print(f"CUDA Operator Benchmark - {datetime.now()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("="*60)

        self.benchmark_reduce()
        self.benchmark_softmax()
        self.benchmark_norm()
        self.benchmark_matmul()

        return self.results

    def save_results(self, filename=None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"

        output = {
            'timestamp': datetime.now().isoformat(),
            'gpu': torch.cuda.get_device_name(0),
            'results': self.results
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {filename}")


def main():
    runner = BenchmarkRunner(num_warmup=10, num_iterations=100)
    runner.run_all()
    runner.save_results()


if __name__ == "__main__":
    main()
