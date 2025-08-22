#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "u256.cuh"

__global__ void benchmark_u256_operations(u64* results, int num_operations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_operations) return;

    u64 a[4], b[4], r[4];
    
    for (int i = 0; i < 4; i++) {
        a[i] = idx + i + 1; // +1 çâ®¡ë ¨§¡¥¦ âì ¤¥«¥­¨ï ­  ­®«ì
        b[i] = idx * 2 + i + 1;
    }
    
    u256Add(r, a, b);
    u256Mul(r, a, b);
    u256Div(r, a, b);
    
    for (int i = 0; i < 4; i++) {
        results[idx * 4 + i] = r[i];
    }
}

int main() {
    const int num_operations = 1000000;
    const int block_size = 256;
    const int grid_size = (num_operations + block_size - 1) / block_size;
    
    u64* d_results;
    cudaMalloc(&d_results, num_operations * 4 * sizeof(u64));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    benchmark_u256_operations<<<grid_size, block_size>>>(d_results, num_operations);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "num_operations: " << num_operations << " at " << milliseconds << " ms" << std::endl;
    std::cout << "benchmark: " << (num_operations / milliseconds) << " op/ms" << std::endl;
    
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

