/**
 * pybind11 Python 绑定 - 将 CUDA 算子暴露给 Python
 *
 * 学习目标：
 * 1. 理解如何使用 pybind11 绑定 C++/CUDA 函数到 Python
 * 2. 学习 numpy 数组与 CUDA 指针的转换
 * 3. 掌握内存管理（分配/释放设备内存）
 * 4. 理解 Python API 设计原则
 *
 * 架构说明：
 * - Python 调用 -> pybind11 封装 -> CUDA Host 函数 -> CUDA Kernel
 * - 所有内存分配/释放都在 C++ 层完成，对 Python 透明
 * - 使用 numpy 数组作为数据交换格式（零拷贝或拷贝）
 */

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cstring>

#include "ops/reduce.h"
#include "ops/softmax.h"
#include "ops/layernorm.h"
#include "ops/rmsnorm.h"
#include "ops/matmul.h"
#include "common/cuda_utils.h"

namespace py = pybind11;

// 在包含头文件后定义命名空间别名
namespace softmax_ops = softmax;
namespace layernorm_ops = layernorm;
namespace rmsnorm_ops = rmsnorm;
namespace matmul_ops = matmul;
namespace reduce_ops = reduce;

/**
 * 从 numpy 数组获取原始指针
 *
 * py::array_t<float> 是 pybind11 的 numpy 数组包装类
 * .data() 返回指向数组数据的指针
 */
float* get_ptr(py::array_t<float>& arr) {
    return arr.mutable_data();
}

/**
 * 检查数组是否 C-连续
 *
 * 重要性：CUDA kernel 假设数据是行优先、连续的
 * 非连续数组（如切片）需要拷贝或特殊处理
 */
void check_contiguous(py::array_t<float>& arr) {
    if (!(arr.flags() & py::array::c_style)) {
        throw std::runtime_error("Array must be C-contiguous");
    }
}

/**
 * Reduce Sum 包装函数
 *
 * 流程：
 * 1. 检查输入合法性（连续）
 * 2. 分配设备内存
 * 3. 拷贝输入到设备（H2D）
 * 4. 启动 CUDA kernel
 * 5. 同步等待完成
 * 6. 拷贝结果回主机（D2H）
 * 7. 释放设备内存
 *
 * impl 参数允许用户选择不同实现进行对比测试
 */
py::array_t<float> reduce_sum(py::array_t<float> input, const std::string& impl = "auto") {
    check_contiguous(input);

    int n = input.size();
    // 创建输出数组（大小为 1）
    auto output = py::array_t<float>(1);
    output.mutable_at(0) = 0.0f;

    const float* d_input = input.data();
    float* d_output = output.mutable_data();

    // 分配设备内存
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    // H2D 拷贝：将 numpy 数据传到 GPU
    CUDA_CHECK(cudaMemcpy(d_in, d_input, n * sizeof(float), cudaMemcpyHostToDevice));
    // 初始化输出为 0（累加需要初始值）
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    // 根据 impl 参数选择不同实现
    if (impl == "naive") {
        reduce::reduce_sum_naive(d_in, d_out, n);
    } else if (impl == "shared") {
        reduce::reduce_sum_shared(d_in, d_out, n);
    } else if (impl == "warp") {
        reduce::reduce_sum_warp(d_in, d_out, n);
    } else if (impl == "twopass") {
        reduce::reduce_sum_twopass(d_in, d_out, n);
    } else {
        reduce::reduce_sum(d_in, d_out, n);  // auto
    }

    // 同步确保 kernel 完成
    CUDA_CHECK(cudaDeviceSynchronize());
    // D2H 拷贝：结果传回 CPU
    CUDA_CHECK(cudaMemcpy(d_output, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    // 释放设备内存（重要！避免内存泄漏）
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return output;
}

/**
 * Softmax 包装函数
 *
 * 需要显式传入 rows 和 cols，因为 numpy 数组可能是展平的
 * 这是为了与 CUDA kernel 的接口保持一致
 */
py::array_t<float> softmax(py::array_t<float> input, int rows, int cols,
                           const std::string& impl = "auto") {
    check_contiguous(input);

    // 输出数组与输入形状相同
    auto output = py::array_t<float>(input.size());

    const float* d_input = input.data();
    float* d_output = output.mutable_data();

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, rows * cols * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, d_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    if (impl == "naive") {
        softmax_ops::softmax_naive(d_in, d_out, rows, cols);
    } else if (impl == "online") {
        softmax_ops::softmax_online(d_in, d_out, rows, cols);
    } else if (impl == "warp") {
        softmax_ops::softmax_warp(d_in, d_out, rows, cols);
    } else {
        softmax_ops::softmax(d_in, d_out, rows, cols);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_output, d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    // 重塑形状以匹配输入
    output.resize({rows, cols});
    return output;
}

/**
 * LayerNorm 包装函数
 *
 * weight 和 bias 是可学习的参数，形状为 [cols]
 */
py::array_t<float> layernorm(py::array_t<float> input,
                             py::array_t<float> weight,
                             py::array_t<float> bias,
                             int rows, int cols,
                             float eps = 1e-5f,
                             const std::string& impl = "auto") {
    check_contiguous(input);
    check_contiguous(weight);
    check_contiguous(bias);

    auto output = py::array_t<float>(rows * cols);

    const float* d_input = input.data();
    const float* d_weight = weight.data();
    const float* d_bias = bias.data();
    float* d_output = output.mutable_data();

    float *d_in, *d_w, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, rows * cols * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, d_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, d_weight, cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, d_bias, cols * sizeof(float), cudaMemcpyHostToDevice));

    if (impl == "naive") {
        layernorm_ops::layernorm_naive(d_in, d_w, d_b, d_out, rows, cols, eps);
    } else if (impl == "warp") {
        layernorm_ops::layernorm_warp(d_in, d_w, d_b, d_out, rows, cols, eps);
    } else if (impl == "vectorized") {
        layernorm_ops::layernorm_vectorized(d_in, d_w, d_b, d_out, rows, cols, eps);
    } else {
        layernorm_ops::layernorm(d_in, d_w, d_b, d_out, rows, cols, eps);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_output, d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    output.resize({rows, cols});
    return output;
}

/**
 * RMSNorm 包装函数
 *
 * 比 LayerNorm 少一个 bias 参数
 */
py::array_t<float> rmsnorm(py::array_t<float> input,
                           py::array_t<float> weight,
                           int rows, int cols,
                           float eps = 1e-5f,
                           const std::string& impl = "auto") {
    check_contiguous(input);
    check_contiguous(weight);

    auto output = py::array_t<float>(rows * cols);

    const float* d_input = input.data();
    const float* d_weight = weight.data();
    float* d_output = output.mutable_data();

    float *d_in, *d_w, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, rows * cols * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, d_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, d_weight, cols * sizeof(float), cudaMemcpyHostToDevice));

    if (impl == "naive") {
        rmsnorm_ops::rmsnorm_naive(d_in, d_w, d_out, rows, cols, eps);
    } else if (impl == "warp") {
        rmsnorm_ops::rmsnorm_warp(d_in, d_w, d_out, rows, cols, eps);
    } else if (impl == "vectorized") {
        rmsnorm_ops::rmsnorm_vectorized(d_in, d_w, d_out, rows, cols, eps);
    } else {
        rmsnorm_ops::rmsnorm(d_in, d_w, d_out, rows, cols, eps);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_output, d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_out));

    output.resize({rows, cols});
    return output;
}

/**
 * MatMul 包装函数
 *
 * 注意：A 的形状是 [M, K]，B 的形状是 [K, N]
 * 输出 C 的形状是 [M, N]
 */
py::array_t<float> matmul(py::array_t<float> A,
                          py::array_t<float> B,
                          int M, int N, int K,
                          const std::string& impl = "auto") {
    check_contiguous(A);
    check_contiguous(B);

    auto C = py::array_t<float>(M * N);

    const float* d_A = A.data();
    const float* d_B = B.data();
    float* d_C = C.mutable_data();

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, d_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, d_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    if (impl == "naive") {
        matmul_ops::matmul_naive(d_a, d_b, d_c, M, N, K);
    } else if (impl == "shared") {
        matmul_ops::matmul_shared(d_a, d_b, d_c, M, N, K);
    } else if (impl == "1d_tiling") {
        matmul_ops::matmul_1d_tiling(d_a, d_b, d_c, M, N, K);
    } else if (impl == "2d_tiling") {
        matmul_ops::matmul_2d_tiling(d_a, d_b, d_c, M, N, K);
    } else {
        matmul_ops::matmul(d_a, d_b, d_c, M, N, K);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_C, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    C.resize({M, N});
    return C;
}

/**
 * 模块定义
 *
 * PYBIND11_MODULE 宏创建 Python 模块
 * m.def() 将 C++ 函数绑定到 Python 名字
 *
 * 参数说明：
 * - 第一个参数：Python 中的函数名
 * - 第二个参数：C++ 函数指针
 * - 第三个参数：文档字符串
 * - py::arg()：参数名和默认值
 */
PYBIND11_MODULE(cuda_ops, m) {
    m.doc() = "CUDA operator development library";

    m.def("reduce_sum", &reduce_sum,
          "Reduce sum operation",
          py::arg("input"), py::arg("impl") = "auto");

    m.def("softmax", static_cast<py::array_t<float>(*)(py::array_t<float>, int, int, const std::string&)>(&softmax),
          "Softmax operation",
          py::arg("input"), py::arg("rows"), py::arg("cols"),
          py::arg("impl") = "auto");

    m.def("layernorm", static_cast<py::array_t<float>(*)(py::array_t<float>, py::array_t<float>, py::array_t<float>, int, int, float, const std::string&)>(&layernorm),
          "LayerNorm operation",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("rows"), py::arg("cols"), py::arg("eps") = 1e-5f,
          py::arg("impl") = "auto");

    m.def("rmsnorm", static_cast<py::array_t<float>(*)(py::array_t<float>, py::array_t<float>, int, int, float, const std::string&)>(&rmsnorm),
          "RMSNorm operation",
          py::arg("input"), py::arg("weight"),
          py::arg("rows"), py::arg("cols"), py::arg("eps") = 1e-5f,
          py::arg("impl") = "auto");

    m.def("matmul", static_cast<py::array_t<float>(*)(py::array_t<float>, py::array_t<float>, int, int, int, const std::string&)>(&matmul),
          "Matrix multiplication C = A @ B",
          py::arg("A"), py::arg("B"), py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("impl") = "auto");

    // 版本信息
    m.attr("__version__") = "0.1.0";
}
