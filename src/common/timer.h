#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <vector>

// CPU timer using chrono
class CPUTimer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }

    void stop() { stop_ = std::chrono::high_resolution_clock::now(); }

    float elapsed_ms() const {
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop_ -
                                                                  start_);
        return duration.count() / 1000.0f;
    }

private:
    std::chrono::high_resolution_clock::time_point start_, stop_;
};

// GPU timer using CUDA events
class GPUTimer {
public:
    GPUTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
    }

    float elapsed_ms() const {
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// Benchmark result structure
struct BenchmarkResult {
    std::string name;
    float avg_ms = 0;
    float min_ms = 0;
    float max_ms = 0;
    float std_ms = 0;
    float gflops = 0;
    float bandwidth_gb_s = 0;
    float utilization_percent = 0;

    void print() const;
};

// Simple benchmark runner
template <typename Func>
BenchmarkResult benchmark(const std::string &name, Func &&f,
                          int warmup = 10, int iterations = 100) {
    GPUTimer timer;
    std::vector<float> times;
    times.reserve(iterations);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        f();
    }
    cudaDeviceSynchronize();

    // Benchmark
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        f();
        timer.stop();
        times.push_back(timer.elapsed_ms());
    }

    // Compute statistics
    BenchmarkResult result;
    result.name = name;

    float sum = 0, min_t = times[0], max_t = times[0];
    for (float t : times) {
        sum += t;
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }
    result.avg_ms = sum / iterations;
    result.min_ms = min_t;
    result.max_ms = max_t;

    float variance = 0;
    for (float t : times) {
        variance += (t - result.avg_ms) * (t - result.avg_ms);
    }
    result.std_ms = std::sqrt(variance / iterations);

    return result;
}
