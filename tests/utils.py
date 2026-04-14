"""Test utilities for CUDA operators."""
import numpy as np
import torch


def check_allclose(actual, expected, rtol=1e-5, atol=1e-6, name=""):
    """Check if two arrays are close and print diagnostics."""
    actual = np.asarray(actual).flatten()
    expected = np.asarray(expected).flatten()

    diff = np.abs(actual - expected)
    max_diff = np.max(diff)
    max_idx = np.argmax(diff)
    mean_diff = np.mean(diff)

    rel_diff = diff / (np.abs(expected) + 1e-8)
    max_rel_diff = np.max(rel_diff)

    passed = np.allclose(actual, expected, rtol=rtol, atol=atol)

    if passed:
        print(f"  {name}: PASSED")
        print(f"    Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
    else:
        print(f"  {name}: FAILED")
        print(f"    Max diff: {max_diff:.2e} at index {max_idx}")
        print(f"    Max relative diff: {max_rel_diff:.2e}")
        print(f"    Actual: {actual[max_idx]:.6f}, Expected: {expected[max_idx]:.6f}")

    return passed


def generate_random_input(shape, dtype=np.float32, seed=42):
    """Generate random input with fixed seed."""
    np.random.seed(seed)
    return np.random.randn(*shape).astype(dtype)


def generate_uniform_input(shape, low=-1.0, high=1.0, dtype=np.float32, seed=42):
    """Generate uniform random input with fixed seed."""
    np.random.seed(seed)
    return np.random.uniform(low, high, shape).astype(dtype)
