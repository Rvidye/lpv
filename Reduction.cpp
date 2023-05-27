#include <iostream>
#include <omp.h>

#define ARRAY_SIZE 1000000

int main() {
    int arr[ARRAY_SIZE];
    int min, max, sum;
    double avg;

    // Initialize the array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = i + 1;
    }

    // Serial calculations
    min = arr[0];
    max = arr[0];
    sum = 0;

    double serialStartTime = omp_get_wtime();

    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (arr[i] < min)
            min = arr[i];
        if (arr[i] > max)
            max = arr[i];
        sum += arr[i];
    }

    avg = (double)sum / ARRAY_SIZE;

    double serialEndTime = omp_get_wtime();
    double serialTime = serialEndTime - serialStartTime;

    std::cout << "Serial Calculation:" << std::endl;
    std::cout << "Min: " << min << std::endl;
    std::cout << "Max: " << max << std::endl;
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Avg: " << avg << std::endl;
    std::cout << "Time taken (Serial): " << serialTime << " seconds" << std::endl;

    // Parallel calculations
    min = arr[0];
    max = arr[0];
    sum = 0;

    double parallelStartTime = omp_get_wtime();

#pragma omp parallel for reduction(min:min) reduction(max:max) reduction(+:sum)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        //int threadID = omp_get_thread_num();
        //std::cout << "Thread ID: " << threadID << ", Index: " << i << std::endl;

        if (arr[i] < min)
            min = arr[i];
        if (arr[i] > max)
            max = arr[i];
        sum += arr[i];
    }

    avg = (double)sum / ARRAY_SIZE;

    double parallelEndTime = omp_get_wtime();
    double parallelTime = parallelEndTime - parallelStartTime;

    std::cout << "\nParallel Calculation:" << std::endl;
    std::cout << "Min: " << min << std::endl;
    std::cout << "Max: " << max << std::endl;
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Avg: " << avg << std::endl;
    std::cout << "Time taken (Parallel): " << parallelTime << " seconds" << std::endl;

    double speedup = serialTime / parallelTime;
    std::cout << "Speedup: " << speedup << std::endl;

    return 0;
}
