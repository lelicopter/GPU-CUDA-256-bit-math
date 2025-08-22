#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "u256.cuh"

// Кернел для тестирования производительности
__global__ void benchmark_u256_operations(u64* results, int num_operations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_operations) return;

    u64 a[4], b[4], r[4];
    
    // Инициализация тестовых данных
    for (int i = 0; i < 4; i++) {
        a[i] = idx + i + 1; // +1 чтобы избежать деления на ноль
        b[i] = idx * 2 + i + 1;
    }
    
    // Тестирование различных операций
    u256Add(r, a, b);
    u256Mul(r, a, b);
    u256Div(r, a, b);
    
    // Сохранение результатов
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
    
    // Запуск бенчмарка
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    benchmark_u256_operations<<<grid_size, block_size>>>(d_results, num_operations);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Выполнено " << num_operations << " операций за " 
              << milliseconds << " мс" << std::endl;
    std::cout << "Производительность: " << (num_operations / milliseconds) 
              << " операций/мс" << std::endl;
    
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
