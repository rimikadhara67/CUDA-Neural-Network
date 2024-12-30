#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

//brute force approach to finding which point
void findClosestCPU(float3* points, int* indices, int count) {
    // Base case, if there's 1 point don't do anything
    if(count <=1) return;
    // Loop through every point
    for (int curPoint = 0; curPoint < count; curPoint++) {
        // set as close to the largest float possible
        float distToClosest = 3.4028238f ;
        // See how far it is from every other point
        for (int i = 0; i < count; i++) {
            // Don't check distance to itself
            if(i == curPoint) continue;
            float dist_sqr = (points[curPoint].x - points[i].x) *
                (points[curPoint].x - points[i].x) +
                (points[curPoint].y - points[i].y) *
                (points[curPoint].y - points[i].y) +
                (points[curPoint].z - points[i].z) *
                (points[curPoint].z - points[i].z);
            if(dist_sqr < distToClosest) {
                distToClosest = dist_sqr;
                indices[curPoint] = i;
            }
        }
    }
}

int main(){

    //defining parameters
    const int count = 10000;
    int* indexOfClosest = new int[count];
    float3* points = new float3[count];

    //defining random points
    for (int i = 0; i < count; i++){
        points[i].x = (float)(((rand()%10000))-5000);
        points[i].y = (float)(((rand()%10000))-5000);
        points[i].z = (float)(((rand()%10000))-5000);
    }

    long fastest = 1000000000;

    cout << "running brute force nearest neighbor on the CPU..."<<endl;
    for (int i = 0; i <= 10; i++){
        long start = clock();
        findClosestCPU(points, indexOfClosest, count);
        double duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
        cout << "test " << i << " took " << duration << " seconds" <<endl;
    }

    return 0;
}