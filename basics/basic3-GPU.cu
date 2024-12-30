%%writefile findClosestGPU.cu
#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// Brute force implementation, parallelized on the GPU
__global__ void findClosestGPU(float3* points, int* indices, int count) {
    if (count <= 1) return;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        float3 thisPoint = points[idx];
        float smallestSoFar = 3.40282e38f;

        for (int i = 0; i < count; i++) {
            if (i == idx) continue;

            float dist_sqr = (thisPoint.x - points[i].x) *
                             (thisPoint.x - points[i].x) +
                             (thisPoint.y - points[i].y) *
                             (thisPoint.y - points[i].y) +
                             (thisPoint.z - points[i].z) *
                             (thisPoint.z - points[i].z);
            if (dist_sqr < smallestSoFar) {
                smallestSoFar = dist_sqr;
                indices[idx] = i;
            }
        }
    }
}

int main() {
    // Defining parameters
    const int count = 10000;
    int* h_indexOfClosest = new int[count];
    float3* h_points = new float3[count];

    // Defining random points
    for (int i = 0; i < count; i++) {
        h_points[i].x = (float)(((rand() % 10000)) - 5000);
        h_points[i].y = (float)(((rand() % 10000)) - 5000);
        h_points[i].z = (float)(((rand() % 10000)) - 5000);
    }

    // Device pointers
    int* d_indexOfClosest;
    float3* d_points;

    // Allocating memory on the device
    cudaMalloc(&d_indexOfClosest, sizeof(int) * count);
    cudaMalloc(&d_points, sizeof(float3) * count);

    // Copying values from the host to the device
    cudaMemcpy(d_points, h_points, sizeof(float3) * count, cudaMemcpyHostToDevice);

    int threads_per_block = 64;
    cout << "Running brute force nearest neighbor on the GPU..." << endl;
    for (int i = 1; i <= 10; i++) {
        long start = clock();

        findClosestGPU<<<(count / threads_per_block) + 1, threads_per_block>>>(d_points, d_indexOfClosest, count);
        cudaDeviceSynchronize();

        // Copying results from the device to the host
        cudaMemcpy(h_indexOfClosest, d_indexOfClosest, sizeof(int) * count, cudaMemcpyDeviceToHost);

        double duration = (clock() - start) / (double)CLOCKS_PER_SEC;
        cout << "Test " << i << " took " << duration << " seconds" << endl;
    }

    // Freeing device memory
    cudaFree(d_indexOfClosest);
    cudaFree(d_points);

    // Freeing host memory
    delete[] h_indexOfClosest;
    delete[] h_points;

    return 0;
}