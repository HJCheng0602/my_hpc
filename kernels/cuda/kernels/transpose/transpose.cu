#include "common.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <numeric>
#include <cub/cub.cuh>


__global__ void copy_naive(float * __restrict__ d_in, float * __restrict__ d_out, int nX, int nY)
{
    int x = blockIdx.x * 32 + threadIdx.x;  
    int y = blockIdx.y * 32 + threadIdx.y;  
    
    for (int j = 0; j < 32; j += 8) 
    {
        if (x < nX && (y + j) < nY) 
        {
            int index = (y + j) * nX + x;
            d_out[index] = d_in[index];
        }
    }
}

__global__ void copy_float4(const float * __restrict__ d_in, float * __restrict__ d_out, int N)
{
    const float4* d_in_f4  = reinterpret_cast<const float4*>(d_in);
    float4* d_out_f4 = reinterpret_cast<float4*>(d_out);

    int total_f4 = N / 4; 
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < total_f4)
    {
        d_out_f4[tid] = d_in_f4[tid];
    }
}

__global__ void transpose_naive(float * __restrict__ d_in, float * __restrict__ d_out, int nX, int nY)
{
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8) 
    {
        if (x < nX && (y + j) < nY) 
        {
            int index = (y + j) * nX + x;
            int index2 = y + j + x * nY;
            d_out[index] = d_in[index2];
        }
    }
}

__global__ void transpose_naive2(float * __restrict__ d_in, float * __restrict__ d_out, int nX, int nY)
{
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8) 
    {
        if (x < nX && (y + j) < nY) 
        {
            int index = (y + j) * nX + x;
            int index2 = y + j + x * nY;
            d_out[index2] = d_in[index];
        }
    }
}


__global__ void transpose_v0(float * __restrict__ d_in, float * __restrict__ d_out, int nX, int nY)
{
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    __shared__ float sdata[32][32];

    for (int j = 0; j < 32; j += 8) 
    {
        if (x < nX && (y + j) < nY) 
        {
            sdata[threadIdx.y + j][threadIdx.x] = d_in[(y + j) * nX + x];
        }
    }

    __syncthreads();

    int trans_x = blockIdx.y * 32 + threadIdx.x;
    int trans_y = blockIdx.x * 32 + threadIdx.y;

    for (int j = 0; j < 32; j += 8)
    {
        if((trans_y + j) < nX && (trans_x) < nY)
        {
            d_out[(trans_y + j) * nY + trans_x] = sdata[threadIdx.x][threadIdx.y + j];
        }
    }

}


__global__ void transpose_v1(float * __restrict__ d_in, float * __restrict__ d_out, int nX, int nY)
{
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    __shared__ float sdata[32][33];

    for (int j = 0; j < 32; j += 8) 
    {
        if (x < nX && (y + j) < nY) 
        {
            sdata[threadIdx.y + j][threadIdx.x] = d_in[(y + j) * nX + x];
        }
    }

    __syncthreads();

    int trans_x = blockIdx.y * 32 + threadIdx.x;
    int trans_y = blockIdx.x * 32 + threadIdx.y;

    for (int j = 0; j < 32; j += 8)
    {
        if((trans_y + j) < nX && (trans_x) < nY)
        {
            d_out[(trans_y + j) * nY + trans_x] = sdata[threadIdx.x][threadIdx.y + j];
        }
    }
}


__global__ void transpose_v2(float * __restrict__ d_in, float * __restrict__ d_out, int nX, int nY)
{
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    __shared__ float sdata[32][33];
    #pragma unroll
    for (int j = 0; j < 32; j += 8) 
    {
        if (x < nX && (y + j) < nY) 
        {
            sdata[threadIdx.y + j][threadIdx.x] = d_in[(y + j) * nX + x];
        }
    }

    __syncthreads();

    int trans_x = blockIdx.y * 32 + threadIdx.x;
    int trans_y = blockIdx.x * 32 + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < 32; j += 8)
    {
        if((trans_y + j) < nX && (trans_x) < nY)
        {
            d_out[(trans_y + j) * nY + trans_x] = sdata[threadIdx.x][threadIdx.y + j];
        }
    }
}


__global__ void transpose_v3(float * __restrict__ d_in, float * __restrict__ d_out, int nX, int nY)
{
    const float4 * d_in_f4 = reinterpret_cast<const float4*>(d_in);
    float4* d_out_f4 = reinterpret_cast<float4*>(d_out);

    int nX_f4 = nX / 4;
    int nY_f4 = nY / 4;

    __shared__ float sdata[32][33];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int row = tid / 8;
    int col = tid % 8;

    int in_x_f4 = blockIdx.x * 8 + col;
    int in_y = blockIdx.y * 32 + row;

    if (in_x_f4 < nX_f4 && in_y < nY)
    {
        float4 in_val = d_in_f4[in_y * nX_f4 + in_x_f4];

        sdata[col * 4 + 0][row] = in_val.x;
        sdata[col * 4 + 1][row] = in_val.y;
        sdata[col * 4 + 2][row] = in_val.z;
        sdata[col * 4 + 3][row] = in_val.w;
    }

    __syncthreads();

    int out_x_f4 = blockIdx.y * 8 + col;
    int out_y    = blockIdx.x * 32 + row;

    if (out_x_f4 < nY_f4 && out_y < nX)
    {
        float4 out_val;
        out_val.x = sdata[row][col * 4 + 0];
        out_val.y = sdata[row][col * 4 + 1];
        out_val.z = sdata[row][col * 4 + 2];
        out_val.w = sdata[row][col * 4 + 3];

        d_out_f4[out_y * nY_f4 + out_x_f4] = out_val;
    }
}

int main()
{
    const int NX = 16384;
    const int NY = 16384;
    const int N = NX * NY;
    
    const int TILE_DIM = 32; 
    const int BLOCK_ROWS = 8; 

    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    dim3 grid((NX + TILE_DIM - 1) / TILE_DIM, (NY + TILE_DIM - 1) / TILE_DIM, 1);

    float *h_in  = new float[N];
    float *h_ans = new float[N]; 
    float *h_out = new float[N]; 

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            h_in[y * NX + x] = (float)(y * NX + x);
            h_ans[x * NY + y] = h_in[y * NX + x]; 
        }
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer timer;
    const int ITERS = 100;
    float ms;
    const double mem_bytes = 2.0 * N * sizeof(float);

    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float))); 
    timer.Start();
    for (int i = 0; i < ITERS; ++i)
    {
        copy_naive<<<grid, block>>>(d_in, d_out, NX, NY);
    }
    ms = timer.Stop() / ITERS;
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    allclose("Copy Baseline", h_out, h_in, N, ms);
    printf("[Copy Baseline]\t bandwidth: %.2f GB/s\t time: %.3f ms\n", mem_bytes / (ms * 1e-3) / 1e9, ms);


    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float))); 
    timer.Start();
    int block_1d = 256;
    int grid_1d = ((N / 4) + block_1d - 1) / block_1d;
    for (int i = 0; i < ITERS; ++i)
    {
        copy_float4<<<grid_1d, block_1d>>>(d_in, d_out, N);
    }
    ms = timer.Stop() / ITERS;
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    allclose("Copy float4", h_out, h_in, N, ms);
    printf("[Copy float4]\t bandwidth: %.2f GB/s\t time: %.3f ms\n", mem_bytes / (ms * 1e-3) / 1e9, ms);

    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
    timer.Start();
    for (int i = 0; i < ITERS; ++i)
    {
        transpose_naive<<<grid, block>>>(d_in, d_out, NX, NY);
    }
    ms = timer.Stop() / ITERS;
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    allclose("Naive Transpose", h_out, h_ans, N, ms);
    printf("[Naive Transpose]\t bandwidth: %.2f GB/s\t time: %.3f ms\n", mem_bytes / (ms * 1e-3) / 1e9, ms);


    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
    timer.Start();
    for (int i = 0; i < ITERS; ++i)
    {
        transpose_naive2<<<grid, block>>>(d_in, d_out, NX, NY);
    }
    ms = timer.Stop() / ITERS;
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    allclose("Naive2 Transpose", h_out, h_ans, N, ms);
    printf("[Naive2 Transpose]\t bandwidth: %.2f GB/s\t time: %.3f ms\n", mem_bytes / (ms * 1e-3) / 1e9, ms);



    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
    timer.Start();
    for (int i = 0; i < ITERS; ++i)
    {
        transpose_v0<<<grid, block>>>(d_in, d_out, NX, NY);
    }
    ms = timer.Stop() / ITERS;
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    allclose("v0 Transpose", h_out, h_ans, N, ms);
    printf("[v0 Transpose]\t bandwidth: %.2f GB/s\t time: %.3f ms\n", mem_bytes / (ms * 1e-3) / 1e9, ms);


    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
    timer.Start();
    for (int i = 0; i < ITERS; ++i)
    {
        transpose_v1<<<grid, block>>>(d_in, d_out, NX, NY);
    }
    ms = timer.Stop() / ITERS;
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    allclose("v1 Transpose", h_out, h_ans, N, ms);
    printf("[v1 Transpose]\t bandwidth: %.2f GB/s\t time: %.3f ms\n", mem_bytes / (ms * 1e-3) / 1e9, ms);


    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
    timer.Start();
    for (int i = 0; i < ITERS; ++i)
    {
        transpose_v2<<<grid, block>>>(d_in, d_out, NX, NY);
    }
    ms = timer.Stop() / ITERS;
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    allclose("v2 Transpose", h_out, h_ans, N, ms);
    printf("[v2 Transpose]\t bandwidth: %.2f GB/s\t time: %.3f ms\n", mem_bytes / (ms * 1e-3) / 1e9, ms);


    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
    timer.Start();
    for (int i = 0; i < ITERS; ++i)
    {
        transpose_v3<<<grid, block>>>(d_in, d_out, NX, NY);
    }
    ms = timer.Stop() / ITERS;
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    allclose("v3 Transpose", h_out, h_ans, N, ms);
    printf("[v3 Transpose]\t bandwidth: %.2f GB/s\t time: %.3f ms\n", mem_bytes / (ms * 1e-3) / 1e9, ms);


    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_ans;
    delete[] h_out;

    return 0;
}