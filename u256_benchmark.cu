#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "u256.cuh"

// ��୥� ��� ���஢���� �ந�����⥫쭮��
__global__ void benchmark_u256_operations(u64* results, int num_operations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_operations) return;

    u64 a[4], b[4], r[4];
    
    // ���樠������ ��⮢�� ������
    for (int i = 0; i < 4; i++) {
        a[i] = idx + i + 1; // +1 �⮡� �������� ������� �� ����
        b[i] = idx * 2 + i + 1;
    }
    
    // ����஢���� ࠧ����� ����権
    u256Add(r, a, b);
    u256Mul(r, a, b);
    u256Div(r, a, b);
    
    // ���࠭���� १���⮢
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
    
    // ����� ���笠ઠ
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    benchmark_u256_operations<<<grid_size, block_size>>>(d_results, num_operations);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "�믮����� " << num_operations << " ����権 �� " 
              << milliseconds << " ��" << std::endl;
    std::cout << "�ந�����⥫쭮���: " << (num_operations / milliseconds) 
              << " ����権/��" << std::endl;
    
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
