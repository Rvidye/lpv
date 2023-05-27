#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>

// Sequential Bubble Sort
void sequentialBubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1])
                std::swap(arr[j], arr[j + 1]);
        }
    }
}

// Parallel Bubble Sort
void parallelBubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    bool swapped = false; // Declare and initialize outside the parallel region
    #pragma omp parallel
    {
        bool localSwapped = false; // Create a local variable for each thread
        for (int i = 0; i < n - 1; ++i) {
            #pragma omp for
            for (int j = 0; j < n - i - 1; ++j) {
                if (arr[j] > arr[j + 1]) {
                    std::swap(arr[j], arr[j + 1]);
                    localSwapped = true;
                }
            }
            #pragma omp barrier // Synchronize the threads before checking the flag
            #pragma omp master // Only one thread should check the flag
            {
                if (localSwapped)
                    swapped = true;
            }
            #pragma omp barrier // Synchronize the threads after updating the flag
            if (!swapped)
                break;
        }
    }
}



// Sequential Merge Sort
void sequentialMergeSort(std::vector<int>& arr) {
    if (arr.size() <= 1)
        return;

    int mid = arr.size() / 2;
    std::vector<int> left(arr.begin(), arr.begin() + mid);
    std::vector<int> right(arr.begin() + mid, arr.end());

    sequentialMergeSort(left);
    sequentialMergeSort(right);

    std::merge(left.begin(), left.end(), right.begin(), right.end(), arr.begin());
}

// Parallel Merge Sort
void parallelMergeSort(std::vector<int>& arr) {
    if (arr.size() <= 1)
        return;

    int mid = arr.size() / 2;
    std::vector<int> left(arr.begin(), arr.begin() + mid);
    std::vector<int> right(arr.begin() + mid, arr.end());

    #pragma omp parallel sections
    {
        #pragma omp section
        parallelMergeSort(left);
        #pragma omp section
        parallelMergeSort(right);
    }

    std::merge(left.begin(), left.end(), right.begin(), right.end(), arr.begin());
}

// Helper function to print an array
void printArray(const std::vector<int>& arr) {
    for (int i : arr) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

// Function to determine the accuracy of the sort
double determineAccuracy(const std::vector<int>& arr) {
    for (int i = 0; i < arr.size() - 1; ++i) {
        if (arr[i] > arr[i + 1])
            return 0.0;
    }
    return 1.0;
}

int main() {
    int size;
    std::cout << "Enter the size of the array: ";
    std::cin >> size;

    // Create and fill the array with random values
    std::vector<int> arr(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, size);
    for (int& num : arr) {
        num = dis(gen);
    }

    std::cout << "Original array: ";
    printArray(arr);

    std::vector<int> seqArr = arr;
    double startTime = omp_get_wtime();
    sequentialBubbleSort(seqArr);
    double seqBubbleSortTime = omp_get_wtime() - startTime;
    std::cout << "Sequential Bubble Sort:\n";
    //printArray(seqArr);
    

    std::vector<int> parArr = arr;
    startTime = omp_get_wtime();
    parallelBubbleSort(parArr);
    double parBubbleSortTime = omp_get_wtime() - startTime;
    std::cout << "Parallel Bubble Sort:\n";
    //printArray(parArr);
    double bubbleSortAccuracy = determineAccuracy(seqArr);

    seqArr = arr;
    startTime = omp_get_wtime();
    sequentialMergeSort(seqArr);
    double seqMergeSortTime = omp_get_wtime() - startTime;
    std::cout << "Sequential Merge Sort:\n";
    //printArray(seqArr);

    parArr = arr;
    startTime = omp_get_wtime();
    parallelMergeSort(parArr);
    double parMergeSortTime = omp_get_wtime() - startTime;
    std::cout << "Parallel Merge Sort:\n";
    double mergeSortAccuracy = determineAccuracy(parArr);

    std::cout << std::endl;
    std::cout << "Sequential Bubble Sort Time: " << seqBubbleSortTime << " seconds" << std::endl;
    std::cout << "Parallel Bubble Sort Time: " << parBubbleSortTime << " seconds" << std::endl;
    std::cout << "Sequential Merge Sort Time: " << seqMergeSortTime << " seconds" << std::endl;
    std::cout << "Parallel Merge Sort Time: " << parMergeSortTime << " seconds" << std::endl;

    double bubbleSortSpeedup = seqBubbleSortTime / parBubbleSortTime;
    double mergeSortSpeedup = seqMergeSortTime / parMergeSortTime;

    std::cout << std::endl;
    std::cout << "Bubble Sort Accuracy: " << bubbleSortAccuracy * 100 << "%" << std::endl;
    std::cout << "Merge Sort Accuracy: " << mergeSortAccuracy * 100 << "%" << std::endl;
    std::cout << "Parallel Bubble Sort Speedup: " << bubbleSortSpeedup << "x" << std::endl;
    std::cout << "Parallel Merge Sort Speedup: " << mergeSortSpeedup << "x" << std::endl;

    return 0;
}
