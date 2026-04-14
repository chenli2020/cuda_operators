# CUDA算子开发 - Colab使用指南

本指南帮助你在Google Colab环境下快速上手CUDA算子开发和学习。

## 📚 目录

1. [快速开始](#快速开始)
2. [环境配置](#环境配置)
3. [开发流程](#开发流程)
4. [调试技巧](#调试技巧)
5. [性能分析](#性能分析)
6. [常见问题](#常见问题)

## 🚀 快速开始

### 1. 打开Notebook

在Google Colab中打开任意算子的notebook：

- `colab_layernorm.ipynb` - LayerNorm算子（推荐从这里开始）
- `colab_rmsnorm.ipynb` - RMSNorm算子
- `colab_softmax.ipynb` - Softmax算子
- `colab_reduce.ipynb` - Reduce算子
- `colab_matmul.ipynb` - MatMul算子

### 2. 配置GPU运行时

1. 点击菜单：**运行时** → **更改运行时类型**
2. 硬件加速器选择：**T4 GPU**
3. 点击**保存**

### 3. 运行Notebook

从上到下依次运行每个代码单元格：
- 点击单元格左侧的▶️按钮
- 或使用快捷键：`Shift + Enter`

## ⚙️ 环境配置

### GPU信息检查

第一个单元格会自动检测并显示GPU信息：

```python
!nvidia-smi  # 显示GPU信息
!nvcc --version  # 显示CUDA版本
```

**预期输出（T4 GPU）：**
- GPU: Tesla T4
- CUDA Version: 11.x 或 12.x
- Compute Capability: 7.5

### 依赖安装

自动安装必要的Python包：

```bash
!pip install -q torch numpy pybind11
```

**安装时间：** 约30秒

## 🔄 开发流程

### 标准流程

每个算子的notebook都遵循统一的结构：

```
1. 检查GPU环境
   └─ 验证CUDA可用性

2. 安装依赖
   └─ 安装torch, numpy等

3. 创建CUDA源代码
   ├─ cuda_utils.h (工具函数)
   ├─ xxx.cu (CUDA实现)
   └─ binding.cpp (Python绑定)

4. 构建CUDA扩展
   └─ 使用PyTorch JIT编译

5. 精度测试
   └─ 与PyTorch对比验证

6. 性能测试
   └─ 测量执行时间和带宽

7. 性能对比
   └─ 不同实现的性能对比

8. 总结
   └─ 优化技术回顾
```

### 修改和实验

**建议的学习方式：**

1. **第一次运行**：按顺序执行所有单元格，理解完整流程
2. **修改参数**：改变batch size、hidden size等参数，观察性能变化
3. **修改代码**：尝试优化kernel代码，重新编译测试
4. **对比分析**：比较不同实现的性能差异

**示例修改：**

```python
# 修改测试配置
test_cases = [
    (16, 512),   # 改成更小的size
    (32, 1024),
    (64, 2048),
]
```

## 🐛 调试技巧

### 1. CUDA错误检查

如果遇到CUDA错误，启用同步模式：

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

### 2. 查看详细编译信息

编译时使用`verbose=True`：

```python
cuda_ops = load(
    name='my_cuda',
    sources=['...'],
    verbose=True  # 显示详细编译信息
)
```

### 3. 常见错误及解决

**错误1：CUDA out of memory**
```python
# 解决：减小batch size或矩阵维度
input_t = torch.randn(16, 1024, device=device)  # 减小batch size
```

**错误2：Compilation error**
```python
# 解决：检查CUDA语法，确保分号、括号匹配
```

**错误3：Runtime error**
```python
# 解决：检查输入尺寸是否正确
print(f"Input shape: {input_t.shape}")
```

### 4. 调试输出

在CUDA代码中添加调试输出：

```cuda
__device__ void debug_print(const char* msg, float val) {
    if (threadIdx.x == 0) {
        printf("%s: %f\\n", msg, val);
    }
}
```

## 📊 性能分析

### PyTorch Profiler

Colab不支持Nsight，使用PyTorch内置profiler：

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    output = cuda_ops.layernorm(input, weight, bias)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 手动计算带宽利用率

```python
# 计算带宽利用率
bytes_per_elem = 4  # float32
total_bytes = (rows * cols * 3) * bytes_per_elem  # 读3次
bandwidth_gb_s = total_bytes / (time_ms / 1000) / 1e9

# T4峰值带宽320 GB/s
utilization = (bandwidth_gb_s / 320.0) * 100
print(f"Bandwidth: {bandwidth_gb_s:.2f} GB/s ({utilization:.1f}%)")
```

### 性能目标

**在Colab T4上的合理预期：**

| 算子 | 配置 | 目标时间 | 目标带宽 |
|------|------|----------|----------|
| LayerNorm | [32, 4096] | < 0.1 ms | > 200 GB/s |
| RMSNorm | [32, 4096] | < 0.08 ms | > 220 GB/s |
| Softmax | [32, 2048] | < 0.05 ms | > 180 GB/s |
| Reduce | 1M elements | < 0.02 ms | > 250 GB/s |
| MatMul | [1024, 1024, 1024] | < 1 ms | > 500 GFLOPS |

**注意：** Colab上的性能会低于本地A100，重点关注相对性能（加速比）。

## ❓ 常见问题

### Q1: 为什么编译很慢？

**A:** 第一次编译需要1-2分钟，后续会快很多。这是正常的。

### Q2: 可以用其他GPU吗？

**A:** Colab主要提供T4 GPU，免费版也可能分配K80（较老）。付费版可以使用A100/V100。

### Q3: 如何保存我的修改？

**A:** 点击：**文件** → **保存副本到Drive**，或下载.ipynb文件。

### Q4: 运行时间限制？

**A:** 免费版有会话时间限制（约12小时），长时间运行会被断开。建议定期保存结果。

### Q5: 能否使用Nsight？

**A:** Colab不支持Nsight Compute/Systems。请使用PyTorch profiler作为替代。

### Q6: 如何提升性能？

**A:**
1. 确保使用了最新版本的PyTorch
2. 使用Torch JIT编译时的`-O3`优化
3. 检查GPU是否真的在使用（不是CPU fallback）

### Q7: 性能不如预期？

**A:**
1. 检查输入大小是否合适（太小则overhead大）
2. 确保多次迭代取平均
3. 查看是否有warmup（第一次运行总是慢）
4. Colab的T4性能比A100低很多

### Q8: 如何学习优化技巧？

**A:** 建议的学习顺序：
1. 先运行`colab_layernorm.ipynb`，理解基本流程
2. 再运行`colab_reduce.ipynb`，看优化演进
3. 然后学习其他算子
4. 最后尝试自己实现简单算子

## 📖 推荐学习路径

### 第1周：入门
- 运行`colab_layernorm.ipynb`
- 理解CUDA基础概念
- 学习kernel编写

### 第2周：优化技术
- 运行`colab_reduce.ipynb`
- 理解内存合并和共享内存
- 学习warp shuffle

### 第3周：高级算子
- 运行`colab_softmax.ipynb`和`colab_rmsnorm.ipynb`
- 理解数值稳定性
- 学习不同归一化方法

### 第4周：计算密集型
- 运行`colab_matmul.ipynb`
- 理解分块技术
- 学习计算优化

### 第5周：实践
- 使用模板实现简单算子
- 性能分析和优化
- 与PyTorch对比

## 🔗 有用资源

- [PyTorch C++ Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [项目主README](README.md)
- [优化指南](OPTIMIZATION_GUIDE.md)

## 💡 最佳实践

1. **定期保存**：Colab会话可能随时断开，定期保存重要结果
2. **使用GPU**：确保真的在GPU上运行（查看nvidia-smi）
3. **多次测量**：性能测试时多次迭代取平均
4. **对比基准**：始终与PyTorch实现对比
5. **理解原理**：不要只看性能数字，要理解为什么

---

**祝你学习愉快！** 🎉

如有问题，请参考项目主README或提交issue。
