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

// 包装函数：使用不同的函数名避免与命名空间冲突
py::array_t<float> softmax_wrapper(py::array_t<float> input,
                                    int rows, int cols,
                                    const std::string& impl = "auto") {
    check_contiguous(input);

    // 创建输出数组
    py::array_t<float> output = py::array_t<float>(rows * cols);

    const float* d_input = input.data();
    float* d_output = output.mutable_data();

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, rows * cols * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, d_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    if (impl == "naive") {
        softmax::softmax_naive(d_in, d_out, rows, cols);
    } else if (impl == "online") {
        softmax::softmax_online(d_in, d_out, rows, cols);
    } else if (impl == "warp") {
        softmax::softmax_warp(d_in, d_out, rows, cols);
    } else {
        softmax::softmax(d_in, d_out, rows, cols);
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
 */
py::array_t<float> layernorm_wrapper(py::array_t<float> input,
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
        layernorm::layernorm_naive(d_in, d_w, d_b, d_out, rows, cols, eps);
    } else if (impl == "warp") {
        layernorm::layernorm_warp(d_in, d_w, d_b, d_out, rows, cols, eps);
    } else if (impl == "vectorized") {
        layernorm::layernorm_vectorized(d_in, d_w, d_b, d_out, rows, cols, eps);
    } else {
        layernorm::layernorm(d_in, d_w, d_b, d_out, rows, cols, eps);
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
 */
py::array_t<float> rmsnorm_wrapper(py::array_t<float> input,
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
        rmsnorm::rmsnorm_naive(d_in, d_w, d_out, rows, cols, eps);
    } else if (impl == "warp") {
        rmsnorm::rmsnorm_warp(d_in, d_w, d_out, rows, cols, eps);
    } else if (impl == "vectorized") {
        rmsnorm::rmsnorm_vectorized(d_in, d_w, d_out, rows, cols, eps);
    } else {
        rmsnorm::rmsnorm(d_in, d_w, d_out, rows, cols, eps);
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
 */
py::array_t<float> matmul_wrapper(py::array_t<float> A,
                                   py::array_t<float> B,
                                   int M, int N, int K,
                                   const std::string& impl = "auto") {
    check_contiguous(A);
    check_contiguous(B);

    auto C = py::array_t<float>(M * N);

    const float* d_a = A.data();
    const float* d_b = B.data();
    float* d_c = C.mutable_data();

    float *d_a_gpu, *d_b_gpu, *d_c_gpu;
    CUDA_CHECK(cudaMalloc(&d_a_gpu, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_gpu, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c_gpu, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a_gpu, d_a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_gpu, d_b, K * N * sizeof(float), cudaMemcpyHostToDevice));

    if (impl == "naive") {
        matmul::matmul_naive(d_a_gpu, d_b_gpu, d_c_gpu, M, N, K);
    } else if (impl == "shared") {
        matmul::matmul_shared(d_a_gpu, d_b_gpu, d_c_gpu, M, N, K);
    } else if (impl == "1d_tiling") {
        matmul::matmul_1d_tiling(d_a_gpu, d_b_gpu, d_c_gpu, M, N, K);
    } else if (impl == "2d_tiling") {
        matmul::matmul_2d_tiling(d_a_gpu, d_b_gpu, d_c_gpu, M, N, K);
    } else {
        matmul::matmul(d_a_gpu, d_b_gpu, d_c_gpu, M, N, K);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_c, d_c_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a_gpu));
    CUDA_CHECK(cudaFree(d_b_gpu));
    CUDA_CHECK(cudaFree(d_c_gpu));

    C.resize({M, N});
    return C;
}

/**
 * Reduce 包装函数
 */
py::array_t<float> reduce_sum_wrapper(py::array_t<float> input,
                                       const std::string& impl = "auto") {
    check_contiguous(input);

    size_t size = input.size();
    auto output = py::array_t<float>(1);

    const float* d_input = input.data();
    float* d_output = output.mutable_data();

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, d_input, size * sizeof(float), cudaMemcpyHostToDevice));

    if (impl == "naive") {
        reduce::reduce_sum_naive(d_in, d_out, size);
    } else if (impl == "shared") {
        reduce::reduce_sum_shared(d_in, d_out, size);
    } else if (impl == "warp") {
        reduce::reduce_sum_warp(d_in, d_out, size);
    } else {
        reduce::reduce_sum(d_in, d_out, size);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_output, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return output;
}

/**
 * pybind11 模块定义
 *
 * 参数说明：
 * - 第一个参数：Python 中的函数名
 * - 第二个参数：C++ 函数指针
 * - 第三个参数：文档字符串
 * - py::arg()：参数名和默认值
 */
PYBIND11_MODULE(cuda_ops, m) {
    m.doc() = "CUDA operator development library";

    m.def("reduce_sum", &reduce_sum_wrapper,
          "Reduce sum operation",
          py::arg("input"), py::arg("impl") = "auto");

    m.def("softmax", &softmax_wrapper,
          "Softmax operation",
          py::arg("input"), py::arg("rows"), py::arg("cols"),
          py::arg("impl") = "auto");

    m.def("layernorm", &layernorm_wrapper,
          "LayerNorm operation",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("rows"), py::arg("cols"), py::arg("eps") = 1e-5f,
          py::arg("impl") = "auto");

    m.def("rmsnorm", &rmsnorm_wrapper,
          "RMSNorm operation",
          py::arg("input"), py::arg("weight"),
          py::arg("rows"), py::arg("cols"), py::arg("eps") = 1e-5f,
          py::arg("impl") = "auto");

    m.def("matmul", &matmul_wrapper,
          "Matrix multiplication C = A @ B",
          py::arg("A"), py::arg("B"), py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("impl") = "auto");

    // 版本信息
    m.attr("__version__") = "0.1.0";
}
