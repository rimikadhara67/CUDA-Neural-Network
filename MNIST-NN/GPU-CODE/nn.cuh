#include <stdlib.h>
#include <math.h>
#include <random>
#include <algorithm>

#include <cublas_v2.h>
#include <curand.h>

#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "curand.lib")

#include "mnist.cuh"
#include "timer.h"

using namespace std;

#define BLOCK_SIZE 32

__global__ void grad_desc_weights(double *weights, double *gradient, double learning_rate, int rows, int cols){
  //using the blockIDx , blockDim , and threadIDx to access and allocate a computation to a specific thread with respect to its block 
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  int stride-x = gridDim.x * blockDim.x;
  int stride-y = gridDim.y * blockDim.y;

  //this is the same as the normal cpu version but with respect to the stride
  for(int i=idx_x; i<rows; i+=stride_x){
    for(int j=idx_y; j<cols; j+=stride_y){
      weights[i * cols + j] -= learning_rate * gradient[i * cols + j];
    }
  }
}

__global__ void grad_desc_bias(double *bias, double *gradient, double learning_rate, int size){
  //same thing as above but this time we only access the x-dim of our mem-hierarchy
  //TODO: explain why and also the purpose of this function
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  //same as cpu version
  for(int i = idx; i < size; i += stride){
    bias[idx] -= learning_rate * error[idx];
  }
}

__global__ void calc_avgs(double *error, double *avgs, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < rows; i += stride)
    {
        auto sum = 0.;
        for (int b = 0; b < cols; b++)
        {
            sum += error[idx * cols + b];
        }
        avgs[idx] = sum / double(cols);
    }
}

__global__ void sigmoid_derivative(double *y_truth, double *output_layer, double *output_layer_error, int rows, int cols)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int i = idx_x; i < rows; i += stride_x)
    {
        for (int j = idx_y; j < cols; j += stride_y)
        {
            double error_prime = -1 * (y_truth[i * cols + j] - output_layer[i * cols + j]);
            double sigmoid_derivative = output_layer[i * cols + j] * (1 - output_layer[i * cols + j]);
            output_layer_error[i * cols + j] = error_prime * sigmoid_derivative;
        }
    }
}

__global__ void tanh_derivative(double *output_layer_error, double *b, double *hidden_layer, double *hidden_layer_error, int output_size, int rows, int cols)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int i = idx_x; i < rows; i += stride_x)
    {
        for (int j = idx_y; j < cols; j += stride_y)
        {
            double error_prime = 0;
            for (int k = 0; k < output_size; k++)
            {
                error_prime += output_layer_error[k * cols + idx_y] * b[k * rows + idx_x]; // b: 10 * 16
            }
            double tanh_derivative = 1. - (hidden_layer[idx_x * cols + idx_y] * hidden_layer[idx_x * cols + idx_y]);
            hidden_layer_error[idx_x * cols + idx_y] = error_prime * tanh_derivative;
        }
    }
}

__global__ void set_up_y_truth(double *dataset, double *dest, int *indices, int rows, int cols)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int i = idx_x; i < rows; i += stride_x)
    {
        for (int j = idx_y; j < cols; j += stride_y)
        {
            int label = dataset[indices[j]];

            if (i == label)
            {
                dest[i * cols + j] = 1;
            }
            else
            {
                dest[i * cols + j] = 0;
            }
        }
    }
}

__global__ void sigmoid(double *layers, int num, int batch_size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num * batch_size; i += stride)
    {
        auto exp_x = exp(layers[i]);
        layers[i] = exp_x / (exp_x + 1);
    }
}

__global__ void tanh(double *layers, int num, int batch_size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num * batch_size; i += stride)
    {
        auto exp_2x = exp(2 * layers[i]);
        layers[i] = (exp_2x - 1) / (exp_2x + 1);
    }
}

__global__ void add_bias(double *layer, double *bias, int num, int batch_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int row = x; row < num; row += stride_x)
    {
        for (int col = y; col < batch_size; col += stride_y)
        {
            layer[row * batch_size + col] += bias[row];
        }
    }
}

__global__ void transpose(double *mat, double *mat_t, int rows, int cols)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int i = idx_x; i < rows; i += stride_x)
    {
        for (int j = idx_y; j < cols; j += stride_y)
        {
            mat_t[j * rows + i] = mat[i * cols + j];
        }
    }
}

/*
    m - leading dimension of first matrix
    n - Shared dimension of matrices
    o - Trailing dimension of second matrix
*/
__global__ void mat_mul(double *a, double *b, double *c, int m, int n, int o)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int row = x; row < m; row += stride_x)
    {
        for (int col = y; col < o; col += stride_y)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += a[row * n + i] * b[i * o + col];
            }
            c[row * o + col] = sum;
        }
    }
}

// Each column is a data point
__global__ void copy_data(double *dataset, double *dest, int *indices, int rows, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && b < batch_size)
    {
        int data_index = indices[b];
        dest[i * batch_size + b] = dataset[data_index * rows + i];
    }
}

__global__ void init_bias_cuda(double *w, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index != 0)
        return;

    for (int i = 0; i < size; i++)
    {
        w[i] = 0;
    }
}

__global__ void init_weights(double *w, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;


    for (int i = index; i < size; i += stride)
    {
        w[i] = (w[i] * 0.2) - 0.1;
    }
}

void init_weights_cpu(double *w, int rows, int cols)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-0.1, 0.1);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            w[i * cols + j] = dist(gen);
        }
    }
}

//Now, initializing a class for Neural Network
