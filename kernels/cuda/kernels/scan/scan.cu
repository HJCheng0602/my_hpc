#include "common.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <numeric>
#include <cub/cub.cuh>



typedef void (*ScanKernel_t)(const float*, float*, float*, int);

__global__ void add_base_kernel(float *d_out, const float *d_block_sums, int N)
{
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (blockIdx.x > 0 && gid < N) 
    {
        d_out[gid] += d_block_sums[blockIdx.x - 1]; 
    }
}

void dispatch_global_scan(ScanKernel_t kernel_func, int smem_bytes, 
                          const float *d_in, float *d_out, int N) 
{
    const int BLOCK = 1024;
    int GRID = (N + BLOCK - 1) / BLOCK;

    if (GRID == 1) {
        kernel_func<<<1, BLOCK, smem_bytes>>>(d_in, d_out, nullptr, N);
        return;
    }

    float *d_block_sums;
    cudaMalloc(&d_block_sums, GRID * sizeof(float));

    kernel_func<<<GRID, BLOCK, smem_bytes>>>(d_in, d_out, d_block_sums, N);

    dispatch_global_scan(kernel_func, smem_bytes, d_block_sums, d_block_sums, GRID);
    add_base_kernel<<<GRID, BLOCK>>>(d_out, d_block_sums, N);

    cudaFree(d_block_sums);
}


void benchmark_scan(const char* name, ScanKernel_t kernel, int smem_bytes,
                    const float *h_in, const float *h_out, int N, int ITERS = 100) 
{
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    dispatch_global_scan(kernel, smem_bytes, d_in, d_out, N);
    cudaDeviceSynchronize();

    GpuTimer timer;
    timer.Start();
    for (int i = 0; i < ITERS; i++) 
    {
        dispatch_global_scan(kernel, smem_bytes, d_in, d_out, N);
    }
    float ms = timer.Stop() / ITERS;

    float *result = new float[N];
    cudaMemcpy(result, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    allclose(name, result, h_out, N, ms);

    double bandwidth = (2.0 * N * sizeof(float)) / (ms * 1e-3) / 1e9;
    printf("bandwidth: %.2f GB/s\n\n", bandwidth);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] result;
}

__global__ void scan_naive(const float *__restrict__ d_in, float *__restrict__ d_out, int N)
{
    float sum = 0;
    for (int i = 0; i < N; i++)
    {
        sum += d_in[i];
        d_out[i] = sum;
    }
}


// a lot of add operations but the memory bank is good while the 2 * N is not acceptable
__global__ void scan_v0(const float *__restrict__ d_in, float *__restrict__ d_out, float * __restrict__ d_block_sum, int N)
{
    extern __shared__ float temp[]; // allocated on invocation
    int thid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int pout = 0, pin = 1;
    // Load input into shared memory.
    
    temp[pout * blockDim.x + thid] = (gid < N) ? d_in[gid] : 0;
    __syncthreads();
    for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;
        if (thid >= offset)
            temp[pout * blockDim.x + thid] = temp[pin * blockDim.x + thid - offset] + temp[pin * blockDim.x + thid];
        else
            temp[pout * blockDim.x + thid] = temp[pin * blockDim.x + thid];
        __syncthreads();
    }
    if(gid < N)
    {    d_out[gid] = temp[pout * blockDim.x + thid]; // write output
        if(thid == blockDim.x - 1)
            d_block_sum[blockIdx.x] = temp[pout * blockDim.x + thid];
    }
    
}

__global__ void scan_v0_cp(const float *__restrict__ d_in, float *__restrict__ d_out, int N)
{
    extern __shared__ float temp[]; // allocated on invocation
    int thid = threadIdx.x;
    int pout = 0, pin = 1;
    // Load input into shared memory.
    temp[pout * N + thid] = (thid > 0) ?  d_in[thid - 1] : 0;
    __syncthreads();
    for (int offset = 1; offset < N; offset *= 2)
    {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;
        if (thid >= offset)
            temp[pout * N + thid] = temp[pin * N + thid - offset] + temp[pin * N + thid];
        else
            temp[pout * N + thid] = temp[pin * N + thid];
        __syncthreads();
    }
    d_out[thid] = temp[pout * N + thid]; // write output
}

// assume we use it in a block
__global__ void scan_v1(const float * __restrict__ d_in, float * __restrict__ d_out, float * __restrict__ d_block_sums, int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    sdata[tid] = (gid < N) ? d_in[gid] : 0.0f;
    float blocksum;
    __syncthreads();
    for(int d = 1; d < blockDim.x; d <<= 1)
    {
        if((tid + 1) % (d * 2) == 0)
        {
            sdata[tid] = sdata[tid] + sdata[tid - d];
        }
        __syncthreads();
    }

    if(tid == blockDim.x - 1)
    {
        blocksum = sdata[tid];
        sdata[tid] = 0;
    }
    __syncthreads();

    for(int d = blockDim.x / 2; d > 0; d >>= 1)
    {
        if((tid + 1) % (d * 2) == 0)
        {
            float left_val = sdata[tid - d];
            float parent_val = sdata[tid];
            sdata[tid] = left_val + parent_val; 
            sdata[tid - d] = parent_val;        
        }
        __syncthreads();
    }
    if(gid < N)
    {
        if(tid < blockDim.x - 1)
            d_out[gid] = sdata[tid + 1];
        else
            {d_out[gid] = blocksum;
            d_block_sums[blockIdx.x] = blocksum;}
    }
}

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
// reduce warp diverge
__global__ void scan_v2(const float * __restrict__ d_in, float * __restrict__ d_out, float * __restrict__ d_block_sums, int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int pad_tid = tid + CONFLICT_FREE_OFFSET(tid);
    sdata[pad_tid] = (gid < N) ? d_in[gid] : 0.0f;
    float blocksum;
    __syncthreads();
    for(int d = 1; d < blockDim.x; d <<= 1)
    {
        int idx = (tid + 1) * (d * 2) - 1;
        if(idx < blockDim.x)
        {
            int left = idx - d;
            int right = idx;
            
            int pad_left = left + CONFLICT_FREE_OFFSET(left);
            int pad_right = right + CONFLICT_FREE_OFFSET(right);

            sdata[pad_right] += sdata[pad_left];
        }
        __syncthreads();
    }
    if(tid == blockDim.x - 1)
    {
        int root = tid;
        int pad_root = root + CONFLICT_FREE_OFFSET(root);
        blocksum = sdata[pad_root];
        sdata[pad_root] = 0;
    }
    __syncthreads();

    for(int d = blockDim.x / 2; d > 0; d >>= 1)
    {
        int idx = (tid + 1) * (d * 2) - 1;
        if(idx < blockDim.x)
        {
            int pad_left = idx - d + CONFLICT_FREE_OFFSET(idx - d);
            int pad_right = idx + CONFLICT_FREE_OFFSET(idx);

            float left_val = sdata[pad_left];
            float parent_val = sdata[pad_right];
            sdata[pad_right] = left_val + parent_val; 
            sdata[pad_left] = parent_val;        
        }
        __syncthreads();
    }
    if(gid < N)
    {
        if (tid < blockDim.x - 1)
        {
            int next = tid + 1;
            int pad_next = next + CONFLICT_FREE_OFFSET(next);
            d_out[gid] = sdata[pad_next];
        }
        else
        {
            d_out[gid] = blocksum;
            d_block_sums[blockIdx.x] = blocksum;
        }
    }
}
// test in main
int main(void)
{
    // 测试 3200 万规模的任意大小数据 (跨越多 Block 且非 1024 整数倍)
    const int N = 32000013; 

    float *h_in = new float[N];
    float *h_out = new float[N];
    
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;
        sum += h_in[i];
        h_out[i] = sum;
    }

    const int BLOCK = 1024;
    
    int smem_v0 = 2 * BLOCK * sizeof(float);
    int smem_v1 = BLOCK * sizeof(float);    
    int smem_v2 = (BLOCK + CONFLICT_FREE_OFFSET(BLOCK)) * sizeof(float);

    printf("Testing Global Scan with N = %d\n", N);
    printf("----------------------------------------\n");

    benchmark_scan("SCAN v0 (Double Buffer)", scan_v0, smem_v0, h_in, h_out, N);
    benchmark_scan("SCAN v1 (Blelloch Naive)", scan_v1, smem_v1, h_in, h_out, N);
    benchmark_scan("SCAN v2 (Blelloch Pad)",   scan_v2, smem_v2, h_in, h_out, N);

    delete[] h_in;
    delete[] h_out;
    return 0;
}