#include "include/matrix.hh"
#include "include/shape.hh"
#include "include/neural_network.hh"
#include "include/linear_layer.hh"
#include "include/relu_activation.hh"
#include "include/sigmoid_activation.hh"
#include "include/nn_exception.hh"
#include "include/bce_cost.hh"
#include "include/coordinates_dataset.hh"
#include <vector>
#include <ctime> 
#include <iostream> 

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

int main() {
    srand(static_cast<unsigned int>(time(NULL))); // Corrected for proper use of time()

    CoordinatesDataset dataset(100, 21);
    BCECost bce_cost;

    NeuralNetwork nn;
    nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
    nn.addLayer(new ReLUActivation("relu_1"));
    nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
    nn.addLayer(new SigmoidActivation("sigmoid_output"));

    // Network training
    Matrix Y;
    for (int epoch = 0; epoch < 1001; epoch++) {
        float cost = 0.0f;

        for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
            Y = nn.forward(dataset.getBatches().at(batch));
            nn.backprop(Y, dataset.getTargets().at(batch));
            cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch: " << epoch
                      << ", Cost: " << cost / dataset.getNumOfBatches()
                      << std::endl;
        }
    }

    // Compute accuracy
    Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
    Y.copyDeviceToHost();

    float accuracy = computeAccuracy(
        Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
    int m = predictions.shape.x;
    int correct_predictions = 0;

    for (int i = 0; i < m; i++) {
        float prediction = predictions[i] > 0.5f ? 1.0f : 0.0f;
        if (prediction == targets[i]) {
            correct_predictions++;
        }
    }

    return static_cast<float>(correct_predictions) / m;
}
