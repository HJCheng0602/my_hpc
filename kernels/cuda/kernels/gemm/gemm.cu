#include "common.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <numeric>
#include <cub/cub.cuh>
#include <cublas_v2.h>

__global__ void gemm_naive(const float *__restrict__ d_A, const float *__restrict__ d_B, float *__restrict__ d_C,
                           int M, int K, int N)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;

    for (int i = 0; i < K; i++)
    {
        sum += d_A[y_out * K + i] * d_B[i * N + x_out];
    }

    if (x_out < N && y_out < M)
    {
        d_C[y_out * N + x_out] = sum;
    }
}

template <int TILE>
__global__ void gemm_v0(const float *__restrict__ d_A, const float *__restrict__ d_B, float *__restrict__ d_C,
                        int M, int K, int N)
{
    __shared__ float sdata_A[TILE][TILE];
    __shared__ float sdata_B[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0;

    for (int i = 0; i < (K + TILE - 1) / TILE; i++)
    {
        if (row < M && (i * TILE + tx < K))
            sdata_A[ty][tx] = d_A[row * K + i * TILE + tx];
        else
            sdata_A[ty][tx] = 0.0f;
        if (col < N && (i * TILE + ty < K))
            sdata_B[ty][tx] = d_B[(i * TILE + ty) * N + col];
        else
            sdata_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int j = 0; j < TILE; j++)
        {
            sum += sdata_A[ty][j] * sdata_B[j][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        d_C[row * N + col] = sum;
}

template <int TILE>
__global__ void gemm_v1(const float *__restrict__ d_A, const float *__restrict__ d_B, float *__restrict__ d_C,
                        int M, int K, int N)
{
    __shared__ float sdata_A[TILE][TILE + 1];
    __shared__ float sdata_B[TILE][TILE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0;

    for (int i = 0; i < (K + TILE - 1) / TILE; i++)
    {
        if (row < M && (i * TILE + tx < K))
            sdata_A[ty][tx] = d_A[row * K + i * TILE + tx];
        else
            sdata_A[ty][tx] = 0.0f;
        if (col < N && (i * TILE + ty < K))
            sdata_B[ty][tx] = d_B[(i * TILE + ty) * N + col];
        else
            sdata_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int j = 0; j < TILE; j++)
        {
            sum += sdata_A[ty][j] * sdata_B[j][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        d_C[row * N + col] = sum;
}

// every thread is responsible for a 8x8 tile
// while we proccess a 128 x 128 tile in one go, so we only need 16 x 16 size like grid
__global__ void gemm_v2(const float *__restrict__ d_A, const float *__restrict__ d_B, float *__restrict__ d_C,
                        int M, int K, int N)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            int y_out = blockIdx.y * blockDim.y * 8 + threadIdx.y * 8 + j;
            int x_out = blockIdx.x * blockDim.x * 8 + threadIdx.x * 8 + i;
            float sum = 0;
            for (int idx = 0; idx < K; idx++)
            {
                sum += d_A[y_out * K + idx] * d_B[idx * N + x_out];
            }

            if (x_out < N && y_out < M)
            {
                d_C[y_out * N + x_out] = sum;
            }
        }
    }
}

__global__ void gemm_v3(const float *__restrict__ d_A, const float *__restrict__ d_B, float *__restrict__ d_C,
                        int M, int K, int N)
{
    float c_temp[8][8] = {0.0f};

    float a_reg[8];
    float b_reg[8];
    for (int idx = 0; idx < K; idx++)
    {
        for (int j = 0; j < 8; j++)
            a_reg[j] = d_A[(blockIdx.y * blockDim.y * 8 + threadIdx.y * 8 + j) * K + idx];
        for (int i = 0; i < 8; i++)
            b_reg[i] = d_B[idx * N + blockIdx.x * blockDim.x * 8 + threadIdx.x + i * blockDim.x];
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                c_temp[i][j] += a_reg[j] * b_reg[i];
            }
        }
    }
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            int y_out = blockIdx.y * blockDim.y * 8 + threadIdx.y * 8 + j;
            int x_out = blockIdx.x * blockDim.x * 8 + threadIdx.x + i * blockDim.x;
            d_C[y_out * N + x_out] = c_temp[i][j];
        }
    }
}

__global__ void gemm_v4(const float *__restrict__ d_A, const float *__restrict__ d_B, float *__restrict__ d_C,
                        int M, int K, int N)
{
    // the first section is sA
    // while the next section is sB

    __shared__ float ssdata[256][9];
    float c_temp[8][8] = {0.0f};
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * 128;
    int by = blockIdx.y * 128;

    float a_reg[8];
    float b_reg[8];
    for (int idx = 0; idx < K; idx += 8)
    {
        // each thread is responsible for a tile of data
        for (int i = 0; i < 8; i++)
        {
            if (tx + ty * 16 < 128)
            {
                int tid_in = tx + ty * 16;

                int a_row = (tid_in / 8) + i * 16;
                int a_col = tid_in % 8;

                ssdata[a_row][a_col] = d_A[K * (by + a_row) + idx + a_col];
            }
            else
            {
                ssdata[tx + ty * 16][i] = d_B[N * (idx + i) + bx + tx + ty * 16 - 128];
            }
        }

        __syncthreads();
        for (int smallidx = 0; smallidx < 8; smallidx++)
        {
            for (int i = 0; i < 8; i++)
                a_reg[i] = ssdata[ty * 8 + i][smallidx];
            for (int i = 0; i < 8; i++)
                b_reg[i] = ssdata[128 + i * 16 + tx][smallidx];

            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    c_temp[i][j] += a_reg[j] * b_reg[i];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            int y_out = blockIdx.y * blockDim.y * 8 + threadIdx.y * 8 + j;
            int x_out = blockIdx.x * blockDim.x * 8 + threadIdx.x + i * blockDim.x;
            d_C[y_out * N + x_out] = c_temp[i][j];
        }
    }
}

// double buffer method
__global__ void gemm_v5(const float *__restrict__ d_A, const float *__restrict__ d_B, float *__restrict__ d_C,
                        int M, int K, int N)
{
    __shared__ float ssdata[2][256][9];
    float c_temp[8][8] = {0.0f};
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * 128;
    int by = blockIdx.y * 128;

    float a_reg[8];
    float b_reg[8];

    float load_reg[8] = {0};

    int tid_in = tx + ty * 16;
    bool is_A = tid_in < 128;

    

    for (int i = 0; i < 8; i++) {
        if (is_A) {
            int a_row = (tid_in / 8) + i * 16;
            int a_col = tid_in % 8;
            load_reg[i] = d_A[K * (by + a_row) + 0 + a_col];
        } else {
            load_reg[i] = d_B[N * (0 + i) + bx + tid_in - 128];
        }
    }
    for (int i = 0; i < 8; i++) {
        if (is_A) {
            int a_row = (tid_in / 8) + i * 16;
            int a_col = tid_in % 8;
            ssdata[0][a_row][a_col] = load_reg[i];
        } else {
            ssdata[0][tid_in][i] = load_reg[i];
        }
    }
    __syncthreads();

    int write_stage = 1;
    int read_stage = 0;

    for(int idx = 8; idx < K; idx += 8)
    {
        for (int i = 0; i < 8; i++) {
            if (is_A) {
                int a_row = (tid_in / 8) + i * 16;
                int a_col = tid_in % 8;
                load_reg[i] = d_A[K * (by + a_row) + idx + a_col];
            } else {
                load_reg[i] = d_B[N * (idx + i) + bx + tid_in - 128];
            }
        }

        for (int smallidx = 0; smallidx < 8; smallidx++) {
            for (int i = 0; i < 8; i++) a_reg[i] = ssdata[read_stage][ty * 8 + i][smallidx];
            for (int i = 0; i < 8; i++) b_reg[i] = ssdata[read_stage][128 + i * 16 + tx][smallidx];

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    c_temp[i][j] += a_reg[j] * b_reg[i];
                }
            }
        }

        __syncthreads();

        for (int i = 0; i < 8; i++) {
            if (is_A) {
                int a_row = (tid_in / 8) + i * 16;
                int a_col = tid_in % 8;
                ssdata[write_stage][a_row][a_col] = load_reg[i];
            } else {
                ssdata[write_stage][tid_in][i] = load_reg[i];
            }
        }

        __syncthreads();

        write_stage ^= 1;
        read_stage ^= 1;
    }

    for (int smallidx = 0; smallidx < 8; smallidx++) {
        for (int i = 0; i < 8; i++) a_reg[i] = ssdata[read_stage][ty * 8 + i][smallidx];
        for (int i = 0; i < 8; i++) b_reg[i] = ssdata[read_stage][128 + i * 16 + tx][smallidx];

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                c_temp[i][j] += a_reg[j] * b_reg[i];
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int y_out = blockIdx.y * blockDim.y * 8 + threadIdx.y * 8 + j;
            int x_out = blockIdx.x * blockDim.x * 8 + threadIdx.x + i * blockDim.x;
            d_C[y_out * N + x_out] = c_temp[i][j];
        }
    }

}
int main()
{
    const int M = 4096, N = 4096, K = 4096;
    const double flops = 2.0 * M * N * K;
    const double mem_bytes = (M * K + K * N + M * N) * sizeof(float);

    float *hA = new float[M * K];
    float *hB = new float[K * N];
    float *hC = new float[M * N];
    float *hRef = new float[M * N];

    srand(42);
    for (int i = 0; i < M * K; ++i)
        hA[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; ++i)
        hB[i] = (float)rand() / RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer timer;
    const int ITERS = 20;
    float ms;

    printf("%-25s %10s %10s %10s\n", "Kernel", "Time(ms)", "TFLOPS", "BW(GB/s)");
    printf("%s\n", std::string(60, '-').c_str());

    auto bench = [&](const char *name, auto kernel_fn)
    {
        CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));
        kernel_fn(); // warmup
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.Start();
        for (int i = 0; i < ITERS; ++i)
            kernel_fn();
        ms = timer.Stop() / ITERS;

        CUDA_CHECK(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        allclose(name, hC, hRef, M * N, ms);
        printf("%-25s %10.3f %10.3f %10.2f\n",
               name, ms,
               flops / (ms * 1e-3) / 1e12,
               mem_bytes / (ms * 1e-3) / 1e9);
    };

    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.f, beta = 0.f;

        CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));
        // cuBLAS 是 column-major：传 B,A 顺序等价于 row-major 的 A*B
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        CUDA_CHECK(cudaMemcpy(hRef, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        // benchmark cuBLAS
        timer.Start();
        for (int i = 0; i < ITERS; ++i)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        ms = timer.Stop() / ITERS;
        printf("%-25s %10.3f %10.3f %10.2f\n",
               "[cuBLAS baseline]", ms,
               flops / (ms * 1e-3) / 1e12,
               mem_bytes / (ms * 1e-3) / 1e9);
        printf("%s\n", std::string(60, '-').c_str());

        cublasDestroy(handle);
    }

    // ── 各版本 kernel ─────────────────────────────────────────────
    constexpr int TILE = 32;
    dim3 block_2d(TILE, TILE);
    dim3 grid_2d((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    bench("naive", [&]()
          {
        dim3 b(16, 16), g((N+15)/16, (M+15)/16);
        gemm_naive<<<g, b>>>(dA, dB, dC, M, N, K); });

    bench("v0_smem_tile", [&]()
          { gemm_v0<TILE><<<grid_2d, block_2d>>>(dA, dB, dC, M, N, K); });

    bench("v1_smem_tile", [&]()
          { gemm_v1<TILE><<<grid_2d, block_2d>>>(dA, dB, dC, M, N, K); });

    int BN = 128;
    int BM = 128;
    int TN = 8;
    int TM = 8;
    bench("v2_tile", [&]()
          {
        // BM=128, BN=128, BK=8, TM=8, TN=8 → block=(BM/TM, BN/TN)=(16,16)
        dim3 b(BN/TN, BM/TM);
        dim3 g((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_v2<<<g, b>>>(dA, dB, dC, M, N, K); });

    bench("v3_reg", [&]()
          {
        // BM=128, BN=128, BK=8, TM=8, TN=8 → block=(BM/TM, BN/TN)=(16,16)
        dim3 b(BN/TN, BM/TM);
        dim3 g((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_v3<<<g, b>>>(dA, dB, dC, M, N, K); });

    bench("v4_reg_sharedmemory", [&]()
          {
        // BM=128, BN=128, BK=8, TM=8, TN=8 → block=(BM/TM, BN/TN)=(16,16)
        dim3 b(BN/TN, BM/TM);
        dim3 g((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_v4<<<g, b>>>(dA, dB, dC, M, N, K); });

    bench("v5_double_buffer", [&]()
          {
        // BM=128, BN=128, BK=8, TM=8, TN=8 → block=(BM/TM, BN/TN)=(16,16)
        dim3 b(BN/TN, BM/TM);
        dim3 g((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_v5<<<g, b>>>(dA, dB, dC, M, N, K); });

    // bench("v4_double_buffer", [&]() { ... });  // TODO

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;
    delete[] hRef;
    return 0;
}