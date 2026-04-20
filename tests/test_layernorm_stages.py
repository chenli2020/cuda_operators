"""
LayerNorm Stage 1-3 正确性验证

测试新增的优化版本是否与 PyTorch 结果一致
"""

import numpy as np
import torch
import sys
import os

# 使用与其他测试文件一致的导入方式
try:
    import cuda_ops
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
    import cuda_ops


def test_layernorm_stage(impl, rows, cols, atol=1e-4):
    """测试单个实现"""
    # 准备测试数据
    np.random.seed(42)
    input_np = np.random.randn(rows, cols).astype(np.float32)
    weight_np = np.random.randn(cols).astype(np.float32)
    bias_np = np.random.randn(cols).astype(np.float32)

    # PyTorch 参考实现
    input_torch = torch.from_numpy(input_np).cuda()
    weight_torch = torch.from_numpy(weight_np).cuda()
    bias_torch = torch.from_numpy(bias_np).cuda()

    with torch.no_grad():
        output_torch = torch.nn.functional.layer_norm(
            input_torch, (cols,), weight_torch, bias_torch, eps=1e-5
        )
    output_ref = output_torch.cpu().numpy()

    # CUDA 实现
    try:
        output_cuda = cuda_ops.layernorm(input_np, weight_np, bias_np, rows, cols, 1e-5, impl)

        # 比较结果
        max_diff = np.max(np.abs(output_cuda - output_ref))
        mean_diff = np.mean(np.abs(output_cuda - output_ref))

        passed = max_diff < atol

        return passed, max_diff, mean_diff, None
    except Exception as e:
        return False, float('inf'), float('inf'), str(e)


def main():
    print("=" * 80)
    print("LayerNorm Stage 1-3 正确性验证")
    print("=" * 80)
    print()

    # 测试配置
    test_configs = [
        (128, 128, "小矩阵"),
        (128, 512, "中等矩阵"),
        (128, 1024, "中等矩阵 (4的倍数)"),
        (128, 2048, "大矩阵"),
    ]

    # 测试的实现
    implementations = [
        ("naive", "Naive"),
        ("warp", "Warp 优化"),
        ("vectorized", "向量化"),
        ("stage1", "Stage 1: 循环展开+Float8"),
        ("stage2_online", "Stage 2: 在线算法"),
        ("stage2_kahan", "Stage 2: Kahan 求和"),
        ("stage3", "Stage 3: 自动选择"),
        ("stage3_aggressive", "Stage 3: 激进优化"),
        ("stage3_ilp", "Stage 3: 向量化+ILP"),
    ]

    all_passed = True

    for rows, cols, desc in test_configs:
        print(f"\n测试: {desc} (rows={rows}, cols={cols})")
        print("─" * 80)

        for impl, impl_name in implementations:
            # 对于非 4 的倍数的配置，跳过某些实现
            if cols % 8 != 0 and impl in ["stage1"]:
                print(f"{impl_name:30s}: 跳过 (需要 cols 是 8 的倍数)")
                continue

            if cols % 4 != 0 and impl in ["vectorized", "stage2_online", "stage2_kahan", "stage3_ilp"]:
                print(f"{impl_name:30s}: 跳过 (需要 cols 是 4 的倍数)")
                continue

            passed, max_diff, mean_diff, error = test_layernorm_stage(impl, rows, cols)

            if passed:
                print(f"✓ {impl_name:30s}: 最大误差={max_diff:.2e}, 平均误差={mean_diff:.2e}")
            else:
                if error:
                    print(f"✗ {impl_name:30s}: 错误 - {error}")
                else:
                    print(f"✗ {impl_name:30s}: 最大误差={max_diff:.2e} (超过阈值 {1e-4:.2e})")
                all_passed = False

    # 总结
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败，请检查实现")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
