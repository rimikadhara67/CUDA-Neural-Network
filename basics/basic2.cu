#include <stdio.h>

__global__ void hello() {
    printf("Hello from block: (%u, %u, %u), thread: (%u, %u, %u)\n", 
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    // Define the dimensions of the grid and blocks
    dim3 gridDim(2, 2, 2);   // 2x2x2 grid of blocks
    dim3 blockDim(2, 2, 2);  // 2x2x2 grid of threads per block

    // Launch the kernel
    hello<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return 0;
}