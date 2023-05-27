#include <iostream>
#include <vector>
#include <chrono>

// Kernel function for vector addition
__global__ void vectorAddition(float* a, float* b, float* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int size;
    std::cout << "Enter the size of the vectors: ";
    std::cin >> size;

    // Create and initialize the input vectors
    std::vector<float> hostA(size, 1.0);
    std::vector<float> hostB(size, 2.0);
    std::vector<float> hostC(size);

    // Allocate device memory
    float* deviceA;
    float* deviceB;
    float* deviceC;
    cudaMalloc(&deviceA, size * sizeof(float));
    cudaMalloc(&deviceB, size * sizeof(float));
    cudaMalloc(&deviceC, size * sizeof(float));

    // Copy input vectors from host to device
    cudaMemcpy(deviceA, hostA.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // CPU Vector Addition
    auto cpuStartTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; ++i) {
        hostC[i] = hostA[i] + hostB[i];
    }
    auto cpuEndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = cpuEndTime - cpuStartTime;

    // GPU Vector Addition
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuDuration;
    cudaEventElapsedTime(&gpuDuration, start, stop);

    // Copy the result back from device to host
    cudaMemcpy(hostC.data(), deviceC, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    //std::cout << "Result: ";
    //for (int i = 0; i < size; ++i) {
        //std::cout << hostC[i] << " ";
   //}
    std::cout << std::endl;

    // Free device memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    // Print execution times in milliseconds
    std::cout << "CPU Execution Time: " << cpuDuration.count() << " milliseconds" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;

    return 0;
}
