#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ── Error checking ───────────────────────────────────────────────
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// ── Timing ───────────────────────────────────────────────────────
struct GpuTimer
{
    cudaEvent_t start, stop;
    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() { cudaEventRecord(start); }
    float Stop()
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ── 简单正确性检查 ────────────────────────────────────────────────
#include <cmath>
#include <string>


inline bool allclose(const std::string testname, const float *res,
                     const float *ans, int n, float ms, float atol = 1e-4f, float rtol = 1e-4f)
{
    for (int i = 0; i < n; ++i)
    {
        if (std::abs(res[i] - ans[i]) > atol + rtol * std::abs(ans[i]))
        {
            fprintf(stderr, "Mismatch at idx %d: got %.6f, expected %.6f\n",
                    i, res[i], ans[i]);
            fprintf(stderr, "[%s] FAIL  time=%.3f ms\n", testname.c_str(), ms);
            return false;
        }
    }
    fprintf(stderr, "[%s] PASS  time=%.3f ms\n", testname.c_str(), ms);
    

    return true;
}
// ── 常用常量 ──────────────────────────────────────────────────────
constexpr int WARP_SIZE = 32;
