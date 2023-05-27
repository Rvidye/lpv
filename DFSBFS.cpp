#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <omp.h>

// Function to generate a random graph
std::vector<std::vector<int>> generateRandomGraph(int numNodes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, numNodes - 1);

    std::vector<std::vector<int>> graph(numNodes);

    for (int i = 0; i < numNodes; ++i) {
        int numEdges = distribution(gen);
        for (int j = 0; j < numEdges; ++j) {
            int randomNode = distribution(gen);
            if (randomNode != i) {
                graph[i].push_back(randomNode);
            }
        }
    }

    return graph;
}

// Function to perform serial Depth-First Search (DFS)
void serialDFS(std::vector<std::vector<int>>& graph, std::vector<bool>& visited, int node) {
    visited[node] = true;
    std::cout << "Visited node: " << node << std::endl;

    for (int i = 0; i < graph[node].size(); ++i) {
        int adjacentNode = graph[node][i];
        if (!visited[adjacentNode]) {
            serialDFS(graph, visited, adjacentNode);
        }
    }
}

// Function to perform serial Breadth-First Search (BFS)
void serialBFS(std::vector<std::vector<int>>& graph, std::vector<bool>& visited, int startNode) {
    std::queue<int> nodeQueue;
    visited[startNode] = true;
    nodeQueue.push(startNode);

    while (!nodeQueue.empty()) {
        int currentNode = nodeQueue.front();
        nodeQueue.pop();
        std::cout << "Visited node: " << currentNode << std::endl;

        for (int i = 0; i < graph[currentNode].size(); ++i) {
            int adjacentNode = graph[currentNode][i];
            if (!visited[adjacentNode]) {
                visited[adjacentNode] = true;
                nodeQueue.push(adjacentNode);
            }
        }
    }
}

// Function to perform parallel Depth-First Search (DFS)
void parallelDFS(std::vector<std::vector<int>>& graph, std::vector<bool>& visited, int node) {
    visited[node] = true;
    std::cout << "Visited node: " << node << " (Thread ID: " << omp_get_thread_num() << ")" << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < graph[node].size(); ++i) {
        int adjacentNode = graph[node][i];
        if (!visited[adjacentNode]) {
            parallelDFS(graph, visited, adjacentNode);
        }
    }
}

// Function to perform parallel Breadth-First Search (BFS)
void parallelBFS(std::vector<std::vector<int>>& graph, std::vector<bool>& visited, int startNode) {
    std::queue<int> nodeQueue;
    visited[startNode] = true;
    nodeQueue.push(startNode);

    while (!nodeQueue.empty()) {
        int currentNode = nodeQueue.front();
        nodeQueue.pop();
        std::cout << "Visited node: " << currentNode << " (Thread ID: " << omp_get_thread_num() << ")" << std::endl;

        #pragma omp parallel for
        for (int i = 0; i < graph[currentNode].size(); ++i) {
            int adjacentNode = graph[currentNode][i];
            if (!visited[adjacentNode]) {
                #pragma omp critical
                {
                    visited[adjacentNode] = true;
                    nodeQueue.push(adjacentNode);
                }
            }
        }
    }
}

int main() {
    int numNodes;
    std::cout << "Enter the number of nodes in the graph: ";
    std::cin >> numNodes;

    // Generate random graph
    std::vector<std::vector<int>> graph = generateRandomGraph(numNodes);

    int startNode = 0;

    std::vector<bool> visited(numNodes, false);

    // Serial DFS
    auto serialDFSStartTime = std::chrono::high_resolution_clock::now();
    std::cout << "Serial Depth-First Search (DFS):" << std::endl;
    serialDFS(graph, visited, startNode);
    auto serialDFSEndTime = std::chrono::high_resolution_clock::now();

    // Reset visited array for parallel DFS
    std::fill(visited.begin(), visited.end(), false);

    // Serial BFS
    auto serialBFSStartTime = std::chrono::high_resolution_clock::now();
    std::cout << "\nSerial Breadth-First Search (BFS):" << std::endl;
    serialBFS(graph, visited, startNode);
    auto serialBFSEndTime = std::chrono::high_resolution_clock::now();

    // Reset visited array for parallel BFS
    std::fill(visited.begin(), visited.end(), false);

    // Parallel DFS
    auto parallelDFSStartTime = std::chrono::high_resolution_clock::now();
    std::cout << "\nParallel Depth-First Search (DFS):" << std::endl;
    parallelDFS(graph, visited, startNode);
    auto parallelDFSEndTime = std::chrono::high_resolution_clock::now();

    // Reset visited array for parallel BFS
    std::fill(visited.begin(), visited.end(), false);

    // Parallel BFS
    auto parallelBFSStartTime = std::chrono::high_resolution_clock::now();
    std::cout << "\nParallel Breadth-First Search (BFS):" << std::endl;
    parallelBFS(graph, visited, startNode);
    auto parallelBFSEndTime = std::chrono::high_resolution_clock::now();

    // Calculate execution times
    std::chrono::duration<double> serialDFSTime = serialDFSEndTime - serialDFSStartTime;
    std::chrono::duration<double> serialBFSTime = serialBFSEndTime - serialBFSStartTime;
    std::chrono::duration<double> parallelDFSTime = parallelDFSEndTime - parallelDFSStartTime;
    std::chrono::duration<double> parallelBFSTime = parallelBFSEndTime - parallelBFSStartTime;

    std::cout << "\nSerial DFS Execution Time: " << serialDFSTime.count() << " seconds" << std::endl;
    std::cout << "Serial BFS Execution Time: " << serialBFSTime.count() << " seconds" << std::endl;
    std::cout << "Parallel DFS Execution Time: " << parallelDFSTime.count() << " seconds" << std::endl;
    std::cout << "Parallel BFS Execution Time: " << parallelBFSTime.count() << " seconds" << std::endl;

    // Calculate speedup
    double dfsSpeedup = serialDFSTime.count() / parallelDFSTime.count();
    double bfsSpeedup = serialBFSTime.count() / parallelBFSTime.count();

    std::cout << "\nDFS Speedup: " << dfsSpeedup << std::endl;
    std::cout << "BFS Speedup: " << bfsSpeedup << std::endl;

    return 0;
}
