# LayerNorm 优化路线图

## 总览

这个路线图将带你从基础实现逐步优化到接近 cuDNN 性能。

```
Naive → Warp → Vectorized → Online → TensorCore → Fusion → Mixed Precision
```

---

## Stage 0: 基准实现 ✅ (已完成)

**文件**: `src/ops/layernorm.cu`

**已有优化**:
- ✅ Naive 版本（3 遍遍历）
- ✅ Warp 并行归约
- ✅ float4 向量化加载

**性能**: ~50-100 GB/s (A100)

---

## Stage 1: 循环展开 + 更激进的向量化

**优化目标**: 提升 20-30%

**技术要点**:
1. **循环展开** (#pragma unroll)
   - 减少分支预测失败
   - 提高指令级并行

2. **更大的向量加载** (float4 → float4×2)
   - 一次加载 8 个 float
   - 需要 32 字节对齐

3. **寄存器缓存 weight/bias**
   - 减少weight/bias的重复加载

**实现位置**: `layernorm_stage1.cu`

**预期性能**: ~120-150 GB/s

---

## Stage 2: 在线算法 (Online Algorithm)

**优化目标**: 减少 15-20% 的内存访问

**技术要点**:
1. **Welford's Online Algorithm**
   ```
   单遍遍历同时计算 mean 和 variance
   count:      n += 1
   mean:       mean += (x - mean) / n
   variance:   var += (x - mean) * (x - old_mean)
   ```

2. **Kahan Summation**
   - 减少浮点精度损失
   - 补偿误差累加

**优势**:
- 从 2 遍遍历 → 1 遍遍历
- 减少内存访问次数

**实现位置**: `layernorm_stage2.cu`

**预期性能**: ~140-170 GB/s

---

## Stage 3: Warp 级优化 + 寄存器缓存

**优化目标**: 提升 25-35%

**技术要点**:
1. **Warp Shuffle 直接通信**
   - 完全避免 shared memory
   - 使用 `__shfl_down_sync`

2. **寄存器分块**
   - 每个线程缓存多个元素
   - 减少 GMEM 访问

3. **ILP (Instruction Level Parallelism)**
   - 独立计算交错执行
   - 隐藏内存延迟

**实现位置**: `layernorm_stage3.cu`

**预期性能**: ~180-220 GB/s

---

## Stage 4: Tensor Core (WMMA)

**优化目标**: 提升 30-50% (仅大矩阵)

**技术要点**:
1. **Warp Matrix Multiply-Accumulate (WMMA)**
   - 利用 Tensor Core
   - 需要 Ampere/Ada 架构

2. **矩阵分块**
   - 将 reshape 为矩阵运算
   - `mean = ones^T * X / N`

**限制**:
- 仅当 cols ≥ 64 时有效
- 需要 SM70+ (Volta)

**实现位置**: `layernorm_stage4_wmma.cu`

**预期性能**: ~250-300 GB/s (仅大矩阵)

---

## Stage 5: 算子融合 (Kernel Fusion)

**优化目标**: 减少 40-50% 端到端时间

**融合场景**:
1. **Add + LayerNorm**
   ```cuda
   output = layernorm(input + residual)
   ```

2. **Dropout + LayerNorm**
   ```cuda
   output = layernorm(dropout(input))
   ```

3. **Activation + LayerNorm**
   ```cuda
   output = layernorm(gelu(input))
   ```

**优势**:
- 减少 kernel 启动开销
- 减少全局内存往返
- 提高数据局部性

**实现位置**: `layernorm_stage5_fusion.cu`

---

## Stage 6: 混合精度 (Mixed Precision)

**优化目标**: 提升 50-70%

**技术要点**:
1. **FP16 计算**
   - 2x 带宽
   - 2x 计算吞吐
   - Tensor Core 加速

2. **BF16 (A100+)**
   - 更好的动态范围
   - 无需精度调整

3. **量化感知训练**
   - INT8/INT4 推理
   - 需要微调

**实现位置**: `layernorm_stage6_mixed_precision.cu`

---

## 性能对比目标 (A100, rows=4096, cols=4096)

| 版本 | 时间 (μs) | 带宽 (GB/s) | vs PyTorch |
|------|----------|------------|------------|
| PyTorch (native) | 80 | 210 | 1.0x |
| Stage 0 (当前) | 150 | 112 | 0.53x |
| Stage 1 | 120 | 140 | 0.67x |
| Stage 2 | 100 | 168 | 0.80x |
| Stage 3 | 85 | 197 | 0.94x |
| Stage 4 | 70 | 240 | 1.14x |
| Stage 5 (融合) | 50 | 336 | 1.60x |
| Stage 6 (FP16) | 35 | 480 | 2.29x |

---

## 调试工具

### Nsight Compute 分析
```bash
# 查看内存效率
ncu --metrics gld_efficiency,gst_efficiency python tests/test_layernorm.py

# 查看 warp 执行效率
ncu --metrics warp_execution_efficiency,warp_nondivergent_executed python tests/test_layernorm.py

# 查看 Occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_elapsed python tests/test_layernorm.py
```

### 性能剖析
```bash
# 对比不同版本
python benchmark/compare_layernorm_versions.py

# Roofline 分析
python benchmark/roofline_analysis.py
```

---

## 学习建议

1. **先实现 Stage 1-3**
   - 这些是最通用的优化
   - 适用于所有 GPU 架构
   - 代码可读性好

2. **根据场景选择 Stage 4-6**
   - Tensor Core 仅大矩阵
   - 融合需要在具体场景
   - 混合精度需要考虑精度

3. **渐进式优化**
   - 每个阶段都保留前一版本
   - 对比性能和精度
   - 理解每个优化的效果

---

## 参考资源

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Welford's Online Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
- [Tensor Core Programming Guide](https://docs.nvidia.com/cuda/tensor-core-programming-guide/)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
