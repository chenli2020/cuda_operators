# CUDA 算子优化指南 - 从 Naive 到 Expert

## 优化路径总览

```
Naive → Shared Memory → Warp Primitives → Vectorized → Fused
         ↓                    ↓              ↓
      减少全局内存     减少同步开销    提升内存带宽
```

---

## 核心优化技术详解

### 1. Memory Coalescing（内存合并访问）

**问题**：
- GPU 全局内存以 128 字节为单位访问
- 如果 warp 内 32 个线程访问分散的地址，需要多次内存事务

**解决方案**：
- 确保线程 tid 访问地址 `base + tid`（连续）
- 使用向量化加载（float4）一次读取 128 位

**验证方法**：
```bash
ncu --metrics gld_efficiency python test_matmul.py
```

---

### 2. Shared Memory Tiling（共享内存分块）

**原理**：
- 全局内存带宽 (~1-2 TB/s) << 共享内存带宽 (~10+ TB/s)
- 将数据分块加载到共享内存，重复利用

**Reduce 例子**：
```cuda
// 原始：每个元素从全局内存读取 n 次
// 优化：加载到 shared mem，被 block 内所有线程共享
__shared__ float shared[256];
shared[tid] = input[idx];
__syncthreads();
// 现在在 shared mem 上计算
```

**注意点**：
- Bank Conflict：避免多个线程访问同一 bank
- `__syncthreads()`：确保加载完成再计算

---

### 3. Warp Shuffle（Warp 级原语）

**优势**：
- 比 shared memory 更快（寄存器级别交换）
- 不需要 `__syncthreads()`
- 延迟极低

**使用场景**：
- Warp 内归约（reduce、求 max）
- 广播数据

**示例**：
```cuda
// Warp 内求和
float sum = warp_reduce_sum(val);
// 线程 0 获取 warp 总和，然后写入 shared mem
if (lane_id == 0) warp_sums[warp_id] = sum;
```

---

### 4. Vectorized Memory Access（向量化访问）

**条件**：
- 数据对齐到 16 字节边界
- 数据量是 4 的倍数

**效果**：
- 4x 内存带宽利用率
- 减少指令数

**示例**：
```cuda
const float4* in_vec = reinterpret_cast<const float4*>(input);
float4 v = in_vec[idx];  // 一次加载 4 个 float
```

---

### 5. Kernel Fusion（算子融合）

**原理**：
- 合并多个操作到一个 kernel，减少内存往返
- 典型场景：Add + LayerNorm、BiasAdd + Gelu

**RMSNorm 融合例子**：
```cuda
// 融合前：2 个 kernel，2 次全局内存读写
// 1. Add: tmp = input + residual
// 2. RMSNorm: output = rmsnorm(tmp)

// 融合后：1 个 kernel，1 次全局内存读写
__global__ void fused_add_rmsnorm(...) {
    float val = input[i] + residual[i];  // on-the-fly
    // 计算 RMSNorm...
    output[i] = val * inv_rms * weight[i];
}
```

---

## 各算子优化总结

### Reduce Sum

| 版本 | 关键技术 | 全局内存访问次数 | 性能瓶颈 |
|------|---------|-----------------|---------|
| Naive | atomicAdd | O(n) 次原子操作 | 内存争用 |
| Shared | Tree reduction | O(n/block_size) 次原子 | Block 同步 |
| Warp | Shuffle | O(n/32) 次原子 | 最后一个 warp |
| Two-pass | 两级归约 | 2 次遍历 | 2 次 kernel 启动 |

**最佳实践**：
- 小规模 (< 1000)：Shared memory
- 中规模 (< 1M)：Warp shuffle
- 大规模 (> 1M)：Two-pass

---

### Softmax

| 版本 | 关键技术 | 遍历次数 | 数值稳定性 |
|------|---------|---------|-----------|
| Naive | 先求 max 再求 exp | 3 次 | Safe (x - max) |
| Online | Streaming max | 2 次 | Safe |
| Warp | Parallel reduction | 2 次 | Safe |

**关键优化**：
- 必须使用 `x - max` 技巧避免指数爆炸
- Online softmax 减少内存遍历次数

---

### LayerNorm / RMSNorm

| 版本 | 关键技术 | 统计量计算 | 内存带宽 |
|------|---------|-----------|---------|
| Naive | 顺序计算 | 2 遍遍历 | 低 |
| Warp | Parallel reduce | 2 遍并行 | 中 |
| Vectorized | float4 loads | 2 遍并行 | **高** |

**RMSNorm vs LayerNorm**：
- RMSNorm 省去 mean 计算，快 ~15-20%
- LLaMA、Mistral 等现代模型都用 RMSNorm

---

### MatMul

| 版本 | 计算复杂度 | 内存访问 | 优化目标 |
|------|-----------|---------|---------|
| Naive | O(MNK) | O(MNK) | Baseline |
| Shared | O(MNK) | O(MNK/TILE) | 减少 GMEM |
| 1D Tiling | O(MNK) | O(MNK/TILE_A) | A 重用 |
| 2D Tiling | O(MNK) | O(MNK/TILE_AB) | A+B 重用 |

**Roofline 分析**：
- MatMul 是计算密集型（算术强度 = K）
- 当 K > 128 时，性能由计算单元决定（非内存）
- 目标：接近设备峰值 TFLOPS 的 80%

---

## 性能调优 Checklist

### 编译前优化
- [ ] 使用 `__restrict__` 避免指针别名检查
- [ ] 标记 `__forceinline__` 小函数
- [ ] 使用 `--use_fast_math`（如果可以接受精度损失）
- [ ] 选择合适的 CUDA arch（`-arch=sm_80`）

### 内存优化
- [ ] 检查内存访问是否合并（nsight compute）
- [ ] 减少全局内存访问次数（tiling）
- [ ] 使用共享内存缓存频繁访问的数据
- [ ] 考虑向量化加载（float4）

### 并行优化
- [ ] 确保足够的并行度（> 10k 线程）
- [ ] 避免 warp divergence（分支尽量在 warp 级别统一）
- [ ] 使用 warp shuffle 替代 shared memory reduce
- [ ] 优化 block 大小（通常 256 或 512）

### 后处理
- [ ] Kernel fusion 减少内存往返
- [ ] 使用 CUDA Graphs 减少 launch overhead（超大规模）

---

## Nsight 分析命令

### Nsight Compute（详细 kernel 分析）
```bash
# 基本性能指标
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed \
    python tests/test_matmul.py

# 内存效率
ncu --metrics \
    gld_efficiency,gst_efficiency,\
    shared_efficiency \
    python tests/test_reduce.py

# Occupancy（占用率）
ncu --metrics \
    achieved_occupancy,sm__warps_active.avg.pct_of_peak_sustained_elapsed \
    python tests/test_softmax.py
```

### Nsight Systems（系统级分析）
```bash
# 查看 kernel launch 间隔、数据传输
nsys profile -o profile_report python benchmark/benchmark.py
nsys-ui profile_report.nsys-rep  # GUI 查看
```

---

## 性能目标参考（A100）

| 算子 | 目标带宽利用率 | 目标计算利用率 | 典型时间 |
|------|---------------|---------------|---------|
| Reduce | 80-90% | N/A (memory-bound) | 0.1ms @ 1M |
| Softmax | 70-80% | N/A | 0.05ms @ [128, 4096] |
| LayerNorm | 75-85% | N/A | 0.03ms @ [128, 4096] |
| RMSNorm | 80-90% | N/A | 0.025ms @ [128, 4096] |
| MatMul (large) | N/A | 60-80% TF32 | 1ms @ [2048]^3 |

---

## 常见问题与解决

**Q: 精度误差大？**
- 检查数值稳定性（softmax 的 x-max）
- 使用 double 验证，或者与 PyTorch 对比
- 浮点运算顺序不同会导致误差

**Q: 带宽利用率低？**
- 检查内存访问模式（是否合并）
- 使用向量化加载
- 减少不必要的数据搬运（fusion）

**Q: Occupancy 低？**
- 减少寄存器使用（简化代码）
- 减少 shared memory 使用
- 调整 block 大小

**Q: 比 PyTorch 慢？**
- PyTorch 使用 cuDNN/cuBLAS（高度优化）
- 检查是否用了 Tensor Core（WMMA）
- 考虑使用 CUTLASS 库

---

## 学习路径建议

1. **Week 1-2**: 理解 naive 实现，熟悉 CUDA 编程模型
2. **Week 3-4**: 掌握 shared memory 和 tiling
3. **Week 5-6**: 学习 warp shuffle 和向量化
4. **Week 7-8**: 阅读 CUTLASS/FlashAttention 源码，学习生产级优化

---

## 推荐阅读

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/README.md)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
