#pragma once

#include "include/matrix.hh"
#include "include/shape.hh"
#include <vector>
#include <cstdlib>
#include <ctime>

class CoordinatesDataset {
private:
    size_t batch_size;
    size_t number_of_batches;

    std::vector<Matrix> batches; // Specify the type of elements in the vector
    std::vector<Matrix> targets; // Specify the type of elements in the vector

public:
    CoordinatesDataset(size_t batch_size, size_t number_of_batches);

    int getNumOfBatches();
    std::vector<Matrix>& getBatches(); // Return reference to vector of Matrix
    std::vector<Matrix>& getTargets(); // Return reference to vector of Matrix
};