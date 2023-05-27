#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

// CPU Matrix Multiplication
void cpuMatrixMultiplication(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            int sum = 0;
            for (int l = 0; l < n; ++l) {
                sum += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

__global__ void gpuMatrixMultiplication(int* A, int* B, int* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

int main() {
    int m, n, k;
    std::cout << "Enter the size of the matrix (m n k): ";
    std::cin >> m >> n >> k;

    // Initialize matrices A, B, and C
    std::vector<int> h_A(m * n);
    std::vector<int> h_B(n * k);
    std::vector<int> h_C_cpu(m * k);
    std::vector<int> h_C_gpu(m * k);

    // Fill matrices A and B with random values
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < m * n; ++i) {
        h_A[i] = rand() % 100;
    }
    for (int i = 0; i < n * k; ++i) {
        h_B[i] = rand() % 100;
    }

    // Allocate memory on the device
    int* d_A;
    int* d_B;
    int* d_C;
    cudaMalloc((void**)&d_A, m * n * sizeof(int));
    cudaMalloc((void**)&d_B, n * k * sizeof(int));
    cudaMalloc((void**)&d_C, m * k * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_A, h_A.data(), m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), n * k * sizeof(int), cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // Perform matrix multiplication on CPU
    auto cpuStartTime = std::chrono::high_resolution_clock::now();
    cpuMatrixMultiplication(h_A, h_B, h_C_cpu, m, n, k);
    auto cpuEndTime = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication on GPU
    auto gpuStartTime = std::chrono::high_resolution_clock::now();
    gpuMatrixMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    auto gpuEndTime = std::chrono::high_resolution_clock::now();

    // Copy result from device to host
    cudaMemcpy(h_C_gpu.data(), d_C, m * k * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Calculate execution times
    std::chrono::duration<double, std::milli> cpuTime = cpuEndTime - cpuStartTime;
    std::chrono::duration<double, std::milli> gpuTime = gpuEndTime - gpuStartTime;

    // Print the result and execution times
    std::cout << "CPU Execution Time: " << cpuTime.count() << " milliseconds" << std::endl;
    std::cout << "GPU Execution Time: " << gpuTime.count() << " milliseconds" << std::endl;

    // Compare the results
    bool resultsMatch = true;
    for (int i = 0; i < m * k; ++i) {
        if (h_C_cpu[i] != h_C_gpu[i]) {
            resultsMatch = false;
            break;
        }
    }

    std::cout << "Results Match: " << (resultsMatch ? "Yes" : "No") << std::endl;

    return 0;
}
