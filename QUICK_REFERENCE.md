# 快速参考卡片

## 🚀 各平台快速启动

### 云平台（智星云/AutoDL）⭐ 推荐
```bash
git clone <repo>
cd cuda_operators
bash cloud_quickstart.sh
python tests/test_layernorm.py
```
**时间**：5分钟 | **成本**：¥1-2/小时 | **性能**：RTX 3090/4090

### 本地开发
```bash
pip install -e .
python -m pytest tests/
```
**时间**：30分钟 | **成本**：硬件价格 | **性能**：取决于你的 GPU

## 🔧 常用命令

### 环境检查
```bash
# GPU 状态
nvidia-smi

# PyTorch 版本和 CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 编译扩展
python setup.py build_ext --inplace

# 运行测试
python -m pytest tests/ -v
```

### 运行学习脚本
```bash
# 运行单个测试
python tests/test_layernorm.py
python tests/test_softmax.py
python tests/test_reduce.py
python tests/test_rmsnorm.py
python tests/test_matmul.py

# 运行所有测试
python -m pytest tests/ -v

# 运行性能测试
python benchmark/benchmark.py

# 快速学习模式（运行所有测试）
bash run_python_tests.sh      # Linux/Mac
run_python_tests.bat           # Windows
```

### Git 操作
```bash
# 保存进度
git add .
git commit -m "Update code"
git push

# 同步到云平台
git pull
```

## 📊 性能基准

### LayerNorm
| 实现 | 相对性能 |
|------|---------|
| naive | 1.0x |
| warp | 2.5x |
| vectorized | 4.0x |

### Softmax
| 实现 | 相对性能 |
|------|---------|
| naive | 1.0x |
| online | 1.8x |
| warp_optimized | 2.2x |

### MatMul (2048x2048)
| 实现 | 相对性能 |
|------|---------|
| naive | 1.0x |
| shared | 3.5x |
| 2d_tiling | 8.0x |

## ⚡ 性能优化检查清单

- [ ] 使用 warp shuffle 原语
- [ ] 启用向量化加载 (float4)
- [ ] 共享内存优化
- [ ] 减少 bank conflicts
- [ ] 合并全局内存访问
- [ ] 优化占用率 (occupancy)

## 🐛 调试技巧

### CUDA 错误
```bash
# 内存检查
cuda-memcheck python tests/test_layernorm.py

# 同步执行（查看错误行号）
CUDA_LAUNCH_BLOCKING=1 python tests/test_layernorm.py
```

### 性能分析
```bash
# Nsight Compute（kernel 级别）
ncu --kernel-name=layernorm_warp python tests/test_layernorm.py

# Nsight Systems（系统级别）
nsys profile python tests/test_layernorm.py
```

### 内存问题
```bash
# 查看当前内存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Python 内存分析
python -m memory_profiler tests/test_layernorm.py
```

## 💰 成本优化

### 云平台使用技巧
1. **用完即停** - 不用时关机
2. **选择合适GPU** - 学习用 3090
3. **批量处理** - 集中时间完成实验
4. **使用快照** - 保存环境配置

### 预算参考
- **学习阶段** (10小时): ¥15-20
- **项目开发** (50小时): ¥75-100
- **深度优化** (100小时): ¥150-200

## 📚 学习路径

1. **基础** → LayerNorm (理解 CUDA 基础)
2. **进阶** → Softmax (学习优化技巧)
3. **高级** → MatMul (掌握 tiling 优化)
4. **实践** → 自己实现算子

## 🔗 快速链接

- [云平台快速入门](云平台快速入门.md)
- [云平台技术指南](CLOUD_PLATFORM_GUIDE.md)
- [优化技术详解](OPTIMIZATION_GUIDE.md)
- [部署检查清单](DEPLOYMENT_CHECKLIST.md)
- [故障排除](CLOUD_PLATFORM_GUIDE.md#故障排除)

## 🆘 获取帮助

1. **查看文档** - 先搜索相关 FAQ
2. **检查日志** - 查看测试输出
3. **GPU 状态** - `nvidia-smi` 确认 GPU 正常
4. **重新编译** - `python setup.py build_ext --inplace`
5. **重启环境** - 关闭终端，重新运行启动脚本

---

**💡 提示**：遇到问题时，先运行 `bash cloud_quickstart.sh` 重置环境！

## 🎯 推荐工作流程

```bash
# 1. 首次设置
git clone <repo>
cd cuda_operators
bash cloud_quickstart.sh

# 2. 学习某个算子
python tests/test_layernorm.py

# 3. 查看性能对比
python benchmark/benchmark.py

# 4. 修改代码后重新编译
python setup.py build_ext --inplace

# 5. 再次测试
python tests/test_layernorm.py
```

## 💡 代码修改流程

```bash
# 1. 编辑 CUDA 代码
vim src/ops/layernorm.cu

# 2. 重新编译
python setup.py build_ext --inplace

# 3. 测试修改
python tests/test_layernorm.py

# 4. 性能对比
python benchmark/benchmark.py

# 5. 提交代码
git add .
git commit -m "Optimize layernorm"
git push
```
