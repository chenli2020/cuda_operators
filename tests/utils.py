"""
CUDA 算子测试工具函数

学习目标：
1. 理解如何验证浮点数计算的精度
2. 掌握随机数据生成的可复现性
3. 学习误差诊断和分析方法

工具函数说明：
- check_allclose: 比较两个数组的精度
- generate_random_input: 生成正态分布随机数据
- generate_uniform_input: 生成均匀分布随机数据
"""

import numpy as np
import torch


def check_allclose(actual, expected, rtol=1e-5, atol=1e-6, name=""):
    """
    检查两个数组是否接近，并打印详细的诊断信息

    ========================================
    为什么需要这个函数？
    ========================================
    浮点数计算由于精度限制，不同实现（如 CUDA 和 PyTorch）
    可能会产生微小的数值差异。这个函数帮助我们：

    1. 判断差异是否在可接受范围内
    2. 定位最大误差的位置
    3. 分析误差的性质（绝对误差 vs 相对误差）
    4. 提供清晰的测试反馈

    ========================================
    参数说明
    ========================================
    actual: 实际值（通常是 CUDA 实现的结果）
    expected: 期望值（通常是 PyTorch 的结果）
    rtol: 相对误差容忍度（Relative TOLerance）
           公式：|actual - expected| / |expected| <= rtol
           默认 1e-5 表示允许 0.001% 的相对误差
    atol: 绝对误差容忍度（Absolute TOLerance）
           公式：|actual - expected| <= atol
           默认 1e-6 表示允许 0.000001 的绝对误差
    name: 测试名称，用于打印输出

    ========================================
    判断标准
    ========================================
    NumPy 的 allclose 使用混合标准：
    np.abs(actual - expected) <= (atol + rtol * np.abs(expected))

    这意味着：
    - 如果期望值很大（如 1000），主要看相对误差
    - 如果期望值很小（如 0.001），主要看绝对误差
    - 两者结合，避免接近零时的数值问题

    ========================================
    诊断信息说明
    ========================================
    1. Max diff: 最大绝对误差
       - 最大的 |actual - expected|
       - 帮助了解误差的上限

    2. Mean diff: 平均绝对误差
       - 所有误差的平均值
       - 帮助了解整体误差水平

    3. Max relative diff: 最大相对误差
       - 最大的 |actual - expected| / |expected|
       - 帮助了解相对误差的上限

    4. Index: 最大误差的位置
       - 帮助定位问题所在

    5. Actual vs Expected: 对比值
       - 直观看到实际值和期望值的差异

    ========================================
    使用示例
    ========================================
    # 场景 1: 测试 LayerNorm
    actual = cuda_ops.layernorm(...)
    expected = pytorch_layernorm(...)

    # 宽松标准（内存密集型操作）
    check_allclose(actual, expected, rtol=1e-4, atol=1e-5,
                  name="LayerNorm")

    # 严格标准（计算密集型操作）
    check_allclose(actual, expected, rtol=1e-6, atol=1e-7,
                  name="MatMul")

    # 场景 2: 理解误差来源
    if not check_allclose(...):
        # 查看诊断信息，找出问题
        # 可能原因：
        # 1. 浮点运算顺序不同
        # 2. 不同精度（FP32 vs FP16）
        # 3. 实现算法不同
        # 4. 数值稳定性问题
    """
    # ========================================
    # 步骤 1: 展平数组
    # ========================================
    # 无论输入是标量、向量还是矩阵，都转为 1D 数组
    # 这样可以统一处理，避免维度问题
    actual = np.asarray(actual).flatten()
    expected = np.asarray(expected).flatten()

    # ========================================
    # 步骤 2: 计算绝对误差
    # ========================================
    # 逐元素计算绝对误差
    diff = np.abs(actual - expected)

    # 找出最大误差及其位置
    max_diff = np.max(diff)
    max_idx = np.argmax(diff)

    # 计算平均误差（了解整体误差水平）
    mean_diff = np.mean(diff)

    # ========================================
    # 步骤 3: 计算相对误差
    # ========================================
    # 相对误差 = 绝对误差 / |期望值|
    # 加上 1e-8 避免除零
    rel_diff = diff / (np.abs(expected) + 1e-8)

    # 找出最大相对误差
    max_rel_diff = np.max(rel_diff)

    # ========================================
    # 步骤 4: 判断是否通过
    # ========================================
    # 使用 NumPy 的 allclose 函数
    # 它使用混合标准：atol + rtol * |expected|
    passed = np.allclose(actual, expected, rtol=rtol, atol=atol)

    # ========================================
    # 步骤 5: 打印诊断信息
    # ========================================
    if passed:
        # 通过：显示误差统计
        print(f"  {name}: PASSED")
        print(f"    Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
    else:
        # 失败：显示详细的错误信息
        print(f"  {name}: FAILED")
        print(f"    Max diff: {max_diff:.2e} at index {max_idx}")
        print(f"    Max relative diff: {max_rel_diff:.2e}")
        print(f"    Actual: {actual[max_idx]:.6f}, Expected: {expected[max_idx]:.6f}")

        # 额外提示
        if max_rel_diff > 0.01:  # 相对误差 > 1%
            print(f"    [警告] 相对误差较大，可能存在实现问题")
        if max_diff > atol * 10:  # 绝对误差超过容差 10 倍
            print(f"    [警告] 绝对误差较大，可能存在数值稳定性问题")

    return passed


def generate_random_input(shape, dtype=np.float32, seed=42):
    """
    生成固定种子的正态分布随机数据

    ========================================
    为什么需要固定种子？
    ========================================
    1. 可复现性：每次运行生成相同的随机数
    2. 调试方便：发现 bug 时可以重现
    3. 公平对比：不同实现使用相同的输入

    示例：
    # 第一次运行
    data1 = generate_random_input((10, 10))
    # 第二次运行（相同）
    data2 = generate_random_input((10, 10))
    # data1 == data2 (完全相同!)

    ========================================
    为什么使用正态分布？
    ========================================
    1. 真实世界数据大多近似正态分布
       - 人的身高、体重
       - 测量误差
       - 自然现象

    2. 测试覆盖面广
       - 包含正数、负数
       - 接近零的值（测试数值稳定性）
       - 较大的值（测试溢出）

    3. 符合机器学习场景
       - 权重初始化通常用正态分布
       - 输入数据标准化后近似正态分布

    ========================================
    参数说明
    ========================================
    shape: 输出数组的形状
           - 标量：()
           - 向量：(100,)
           - 矩阵：(128, 512)
           - 张量：(32, 3, 224, 224)

    dtype: 数据类型
           - np.float32: 默认，适合 GPU 计算
           - np.float64: 双精度，用于验证
           - np.float16: 半精度，测试混合精度

    seed: 随机种子
          - 42: 常用值（来自《银河系漫游指南》）
          - 任何整数都可以
          - 相同种子 = 相同输出

    ========================================
    使用示例
    ========================================
    # 场景 1: 生成测试数据
    input = generate_random_input((128, 512))
    # 生成 128 行 512 列的数据，值域约在 [-3, 3]

    # 场景 2: 测试不同精度
    data_fp32 = generate_random_input((100, 100), dtype=np.float32)
    data_fp64 = generate_random_input((100, 100), dtype=np.float64)

    # 场景 3: 可复现的测试
    def test_my_function():
        data = generate_random_input((10, 10))
        result1 = my_function(data)
        # 修改代码后...
        result2 = my_function(data)
        # data 相同，可以公平对比 result1 和 result2

    ========================================
    技术细节
    ========================================
    1. np.random.randn(): 生成标准正态分布
       - 均值 μ = 0
       - 标准差 σ = 1
       - 68% 的值在 [-1, 1]
       - 95% 的值在 [-2, 2]
       - 99.7% 的值在 [-3, 3]

    2. .astype(dtype): 转换数据类型
       - 从 float64 (默认) 转为 float32
       - GPU 计算通常使用 float32

    3. 为什么每次都调用 seed？
       - 确保"函数级"的可复现性
       - 避免全局状态的影响
    """
    # 设置随机种子（确保可复现）
    np.random.seed(seed)

    # 生成标准正态分布（均值=0，标准差=1）
    # 然后转换为指定数据类型
    return np.random.randn(*shape).astype(dtype)


def generate_uniform_input(shape, low=-1.0, high=1.0, dtype=np.float32, seed=42):
    """
    生成固定种子的均匀分布随机数据

    ========================================
    正态分布 vs 均匀分布
    ========================================
    正态分布（randn）:
      - 值集中在均值附近
      - 有明显的"中心"
      - 适合模拟真实数据

    均匀分布（uniform）:
      - 值均匀分布在 [low, high]
      - 没有明显的"中心"
      - 适合测试边界情况

    ========================================
    参数说明
    ========================================
    shape: 数组形状（同 generate_random_input）
    low: 均匀分布的下界
         - 默认 -1.0
         - 可设置为任意实数
    high: 均匀分布的上界
          - 默认 1.0
          - 必须 > low
    dtype: 数据类型（同 generate_random_input）
    seed: 随机种子（同 generate_random_input）

    ========================================
    使用场景
    ========================================
    场景 1: 测试激活函数
    # ReLU 在负值区域的行为
    data = generate_uniform_input((100,), low=-2, high=2)
    # 约一半是负数，一半是正数

    场景 2: 测试数值稳定性
    # 生成接近零的值
    data = generate_uniform_input((100,), low=-0.001, high=0.001)
    # 测试除法、log 等操作的稳定性

    场景 3: 测试归一化
    # 生成 [0, 1] 范围的数据
    data = generate_uniform_input((100,), low=0, high=1)
    # 适合测试 softmax、sigmoid 等

    场景 4: 测试极端值
    # 生成很大的值
    data = generate_uniform_input((100,), low=1e6, high=1e7)
    # 测试溢出处理

    ========================================
    技术细节
    ========================================
    1. np.random.uniform(): 均匀分布
       - 每个值的概率相等
       - PDF（概率密度函数）= 1 / (high - low)

    2. 值域分析
       - 理论上：[low, high]
       - 实际上：由于精度限制，可能不包含 high

    3. 与 randn 的选择
       - 一般测试：用 randn（更真实）
       - 边界测试：用 uniform（更可控）
    """
    # 设置随机种子
    np.random.seed(seed)

    # 生成均匀分布
    return np.random.uniform(low, high, shape).astype(dtype)


# ========================================
# 使用建议和最佳实践
# ========================================

"""
1. 何时使用 check_allclose？

   ✅ 适合：
   - 浮点数计算（绝大多数情况）
   - 不同实现的对比（CUDA vs PyTorch）
   - 需要详细诊断信息

   ❌ 不适合：
   - 整数比较（直接用 ==）
   - 布尔值比较（直接用 is）
   - 精确匹配要求（如哈希值）

2. 如何选择 rtol 和 atol？

   宽松标准（快速原型）：
   - rtol=1e-3, atol=1e-4
   - 适合：内存密集型操作（LayerNorm, Softmax）

   标准精度（一般测试）：
   - rtol=1e-5, atol=1e-6
   - 适合：大多数算子

   严格标准（验证精度）：
   - rtol=1e-7, atol=1e-8
   - 适合：计算密集型操作（MatMul）

3. 如何选择随机分布？

   正态分布（randn）：
   - 模拟真实数据
   - 测试一般场景
   - 默认选择

   均匀分布（uniform）：
   - 测试边界情况
   - 特定值域测试
   - 激活函数测试

4. 如何保证可复现性？

   ✅ 好的实践：
   - 固定随机种子
   - 记录种子值
   - 在文档中说明

   ❌ 坏的实践：
   - 不设置种子
   - 使用时间作为种子
   - 依赖全局状态
"""
