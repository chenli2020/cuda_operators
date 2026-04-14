# CUDA Operator Development Template

A minimal, complete CUDA operator development project for learning and benchmarking.

## 🚀 Quick Start Options

### Option 1: Cloud GPU Platforms (智星云/AutoDL) - ⭐ Recommended

**Fast GPU rental with pre-configured environments.**

```bash
# 1. Upload or clone code to platform
git clone <你的仓库地址>
cd cuda_operators

# 2. Run quickstart script
bash cloud_quickstart.sh

# 3. Start learning
python tests/test_layernorm.py
```

📖 **See [CLOUD_PLATFORM_GUIDE.md](CLOUD_PLATFORM_GUIDE.md) for detailed instructions**

### Option 2: Local Development

Requires local GPU and CUDA Toolkit.

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80

# Build
make -j

# Or on Windows with MSVC:
# cmake .. -G "Visual Studio 17 2022" -A x64
# cmake --build . --config Release
```

## Project Structure

```
cuda_operators/
├── CMakeLists.txt          # CMake build configuration
├── pyproject.toml          # Python package configuration
├── src/
│   ├── common/
│   │   ├── cuda_utils.h    # CUDA helper macros and functions
│   │   └── timer.h         # GPU/CPU timing utilities
│   ├── ops/
│   │   ├── reduce.cu/h     # Reduce sum (naive → warp shuffle → two-pass)
│   │   ├── softmax.cu/h    # Softmax (naive → online → warp optimized)
│   │   ├── layernorm.cu/h  # LayerNorm (naive → warp → vectorized)
│   │   ├── rmsnorm.cu/h    # RMSNorm (naive → warp → vectorized)
│   │   └── matmul.cu/h     # Matrix multiplication (naive → shared → 2D tiling)
│   └── binding.cpp         # pybind11 Python bindings
├── tests/
│   ├── test_reduce.py      # Unit tests and benchmarks
│   ├── test_softmax.py
│   ├── test_layernorm.py
│   ├── test_rmsnorm.py
│   └── test_matmul.py
└── benchmark/
    └── benchmark.py        # Comprehensive benchmark suite
```

## Local Build

### Prerequisites

- CUDA Toolkit >= 11.0
- CMake >= 3.18
- Python >= 3.8
- PyTorch (for reference implementation)
- pybind11

### Build

```bash
cd C:\study\cuda_operators

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80

# Build
make -j

# Or on Windows with MSVC:
# cmake .. -G "Visual Studio 17 2022" -A x64
# cmake --build . --config Release
```

### Python Installation

```bash
# Install as Python package
pip install -e .

# Or build wheel
pip wheel .
```

### Run Tests

```bash
# Individual tests
python tests/test_reduce.py
python tests/test_softmax.py
python tests/test_layernorm.py
python tests/test_rmsnorm.py
python tests/test_matmul.py

# Using pytest
pytest tests/

# Run benchmarks
python benchmark/benchmark.py
```

## Platform Comparison

| Feature | Cloud Platforms (智星云/AutoDL) | Local |
|---------|-------------------------------|-------|
| **GPU Required** | ❌ No (rental) | ✅ Yes |
| **Cost** | ~¥1-2/hour | Hardware cost |
| **Setup Time** | ~5 minutes | ~30 minutes |
| **Performance** | RTX 3090/4090 | Your GPU |
| **Persistence** | ✅ Yes | ✅ Yes |
| **Best For** | Serious learning & projects | Production work |

**Recommendation:**
- **Chinese users**: Cloud platforms (智星云/AutoDL) for best speed/price
- **Production**: Local development with Nsight tools

## Operator Implementations

Each operator has multiple implementations showing progressive optimization:

### 1. Reduce Sum
| Implementation | Technique | Complexity |
|---------------|-----------|------------|
| `naive` | Atomic add per thread | O(n) memory traffic |
| `shared` | Shared memory tree reduction | O(n) compute, better coalescing |
| `warp` | Warp shuffle primitives | Reduced synchronization |
| `twopass` | Two-level reduction for large arrays | Scales to any size |

### 2. Softmax
| Implementation | Technique | Key Optimization |
|---------------|-----------|------------------|
| `naive` | Per-thread full row | Baseline |
| `online` | Online softmax algorithm | Numerical stability, single pass |
| `warp` | Warp-level parallel reduction | Better occupancy |

### 3. LayerNorm / RMSNorm
| Implementation | Technique | Key Optimization |
|---------------|-----------|------------------|
| `naive` | Per-thread full row | Baseline |
| `warp` | Warp shuffle for mean/var | Reduced shared mem |
| `vectorized` | float4 loads/stores | 4x memory bandwidth |

### 4. MatMul
| Implementation | Technique | Key Optimization |
|---------------|-----------|------------------|
| `naive` | Global memory only | Baseline |
| `shared` | Shared memory blocking | Reduced global mem traffic |
| `1d_tiling` | Row-wise tile | Better A reuse |
| `2d_tiling` | 2D thread block tile | A and B reuse |

## Usage Example

```python
import numpy as np
import cuda_ops

# Create input
x = np.random.randn(32, 1024).astype(np.float32)

# Run softmax with auto-selected implementation
out = cuda_ops.softmax(x, rows=32, cols=1024, impl="auto")

# Try specific implementations
out_naive = cuda_ops.softmax(x, rows=32, cols=1024, impl="naive")
out_warp = cuda_ops.softmax(x, rows=32, cols=1024, impl="warp")

# MatMul
A = np.random.randn(1024, 512).astype(np.float32)
B = np.random.randn(512, 1024).astype(np.float32)
C = cuda_ops.matmul(A, B, M=1024, N=1024, K=512, impl="2d_tiling")
```

## Performance Profiling with Nsight

### Nsight Compute
```bash
# Profile specific kernel
ncu --kernel-name <kernel_name> --metrics \
    dram__bytes_read.sum,dram__bytes_write.sum,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed \
    python tests/test_matmul.py

# Profile all kernels with summary
ncu --print-summary per-kernel python tests/test_matmul.py
```

### Nsight Systems
```bash
# System-level profiling
nsys profile -o profile_report python tests/test_matmul.py

# View in Nsight Systems GUI
nsys-ui profile_report.nsys-rep
```

### Key Metrics to Track

| Metric | Target | Description |
|--------|--------|-------------|
| Compute Utilization | > 80% | GPU SM utilization |
| Memory Bandwidth | > 80% | Achieved vs theoretical |
| Occupancy | > 70% | Active warps per SM |
| Global Load Efficiency | > 90% | Coalesced memory access |
| Shared Memory Bank Conflicts | < 5% | Avoid bank conflicts |

## Optimization Checklist

When optimizing your kernels:

1. **Memory Coalescing**: Ensure threads in a warp access consecutive memory
2. **Shared Memory**: Use for data reuse, avoid bank conflicts
3. **Warp Primitives**: Use `__shfl_sync` for intra-warp communication
4. **Vectorized Loads**: Use float4 for 4x memory bandwidth
5. **Register Pressure**: Monitor register usage for occupancy
6. **Occupancy**: Balance block size and register/shared mem usage
7. **Tiling**: Maximize data reuse in registers and shared memory

## Expected Performance (A100)

| Operator | Size | Target Bandwidth | Target Compute |
|----------|------|------------------|----------------|
| Reduce | 1M elements | ~80% | Memory-bound |
| Softmax | [128, 4096] | ~70% | Memory-bound |
| LayerNorm | [128, 4096] | ~75% | Memory-bound |
| RMSNorm | [128, 4096] | ~75% | Memory-bound |
| MatMul | [2048, 2048, 2048] | N/A | ~60% TF32 |

## Debugging Tips

1. **cuda-memcheck**: Check for memory errors
   ```bash
   cuda-memcheck python tests/test_matmul.py
   ```

2. **Compute Sanitizer**: Modern alternative to cuda-memcheck
   ```bash
   compute-sanitizer python tests/test_matmul.py
   ```

3. **Verbose Output**: Add CUDA_LAUNCH_BLOCKING for sync errors
   ```bash
   CUDA_LAUNCH_BLOCKING=1 python tests/test_matmul.py
   ```

## Further Reading

- [CLOUD_PLATFORM_GUIDE.md](CLOUD_PLATFORM_GUIDE.md) - Cloud platform usage guide
- [云平台快速入门.md](云平台快速入门.md) - Quick start for Chinese users
- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) - Detailed optimization techniques
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch CUDA Best Practices](https://pytorch.org/tutorials/advanced/cpp_cuda_extension.html)
- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA's CUDA templates

## License

MIT License - Feel free to use for learning and commercial projects.
