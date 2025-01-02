#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <math.h>
#include <random>
#include <algorithm>
#include <chrono>
#include "mnist.h"

class NeuralNetwork
{
public:
    NeuralNetwork();
    void learn(int num_of_epochs);
    void forward(double *input_layer);
    void back_prop(double *y_truth);
    void adjust_weights(double *x);
    int predict(double *input_layer);
    void init_weights(double *w, int rows, int cols);
    void init_bias(double *b, int size);
    void mat_mul(double *a, double *b, double *c, int m, int n, int o);
    void tanh(double *a, int size);
    void sigmoid(double *a, int size);
    double calc_train_accuracy();
    double calc_test_accuracy();

public:
    MNIST mnist;
    double *a;
    double *b;
    double *a_bias;
    double *b_bias;
    double *hidden_layer;
    double *output_layer;
    double *hidden_layer_error;
    double *output_layer_error;
    double learning_rate;
    int input_layer_size;
    int hidden_layer_size;
    int output_layer_size;
    int batch_size;
};

#endif // NEURAL_NETWORK_H
