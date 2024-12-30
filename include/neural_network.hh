#pragma once

#include "include/nn_layer.hh"
#include <vector>
#include "include/bce_cost.hh"
#include "include/matrix.hh" 

class NeuralNetwork {
private:
    std::vector<NNLayer*> layers; // Specify the type of elements in the vector
    BCECost bce_cost;

    Matrix Y;
    Matrix dY;
    float learning_rate;

public:
    NeuralNetwork(float learning_rate = 0.01);
    ~NeuralNetwork();

    Matrix forward(Matrix X);
    void backprop(Matrix predictions, Matrix target);

    void addLayer(NNLayer *layer);
    std::vector<NNLayer*> getLayers() const; /
};