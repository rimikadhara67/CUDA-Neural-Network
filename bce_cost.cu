#include "include/bce_cost.hh"
#include <cmath>
#include <cassert>
#include "include/nn_exception.hh"

__global__ void binaryCrossEntropyCost(float* predictions, float* target, int size, float* cost) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        float pred = predictions[index];
        pred = fmaxf(fminf(pred, 1.0f - 1e-7), 1e-7);

        float partial_cost = target[index] * logf(pred)
                + (1.0f - target[index]) * logf(1.0f - pred);
        atomicAdd(cost, - partial_cost / size);
    }
}

__global__ void dBinaryCrossEntropyCost(float* predictions, float* target, float* dY,
                                        int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        // Clamp predictions to avoid division by zero
        float pred = predictions[index];
        pred = fmaxf(fminf(pred, 1.0f - 1e-7), 1e-7);

        dY[index] = -1.0 * (target[index] / pred - (1 - target[index]) / (1 - pred));
    }
}

//cost (or loss)
float BCECost::cost(Matrix predictions, Matrix target) {
 assert(predictions.shape.x == target.shape.x);

 float* cost;
 cudaMallocManaged(&cost, sizeof(float));
 *cost = 0.0f;

 dim3 block_size(256);
 dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
 binaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
                target.data_device.get(),
                predictions.shape.x, cost);
 cudaDeviceSynchronize();
 NNException::throwIfDeviceErrorsOccurred("Cannot compute binary cross entropy cost.");

 float cost_value = *cost;
 cudaFree(cost);

 return cost_value;
}

//derivative of cost (aka derivative of loss)
Matrix BCECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
 assert(predictions.shape.x == target.shape.x);

 dim3 block_size(256);
 dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
 dBinaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
                 target.data_device.get(),
                 dY.data_device.get(),
                 predictions.shape.x);
 NNException::throwIfDeviceErrorsOccurred("Cannot compute derivative for binary cross entropy.");

 return dY;
}