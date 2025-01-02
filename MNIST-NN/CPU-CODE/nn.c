#include "nn.h"
#include "timer.h"
#include <iostream>

NeuralNetwork::NeuralNetwork() {
    mnist = MNIST();
    mnist.read_data();
    mnist.normalize();

    batch_size = 1;
    input_layer_size = 784;
    hidden_layer_size = 128;
    output_layer_size = 10;

    // Weight matrix for the hidden layer
    a = (double *)malloc(hidden_layer_size * input_layer_size * sizeof(double));
    init_weights(a, hidden_layer_size, input_layer_size);

    a_bias = (double *)malloc(hidden_layer_size * sizeof(double));
    init_bias(a_bias, hidden_layer_size);

    // Weight matrix for the output layer
    b = (double *)malloc(output_layer_size * hidden_layer_size * sizeof(double));
    init_weights(b, output_layer_size, hidden_layer_size);

    b_bias = (double *)malloc(output_layer_size * sizeof(double));
    init_bias(b_bias, output_layer_size);

    hidden_layer = (double *)malloc(hidden_layer_size * batch_size * sizeof(double));
    hidden_layer_error = (double *)malloc(hidden_layer_size * batch_size * sizeof(double));

    output_layer = (double *)malloc(output_layer_size * batch_size * sizeof(double));
    output_layer_error = (double *)malloc(output_layer_size * batch_size * sizeof(double));

    learning_rate = 0.01;
}

void NeuralNetwork::learn(int num_of_epochs) {
    random_device rd;
    mt19937 g(rd());

    vector<int> indices(mnist.train_images.size());
    iota(indices.begin(), indices.end(), 0);

    for (int i = 0; i < num_of_epochs; i++) {
        shuffle(indices.begin(), indices.end(), g);

        for (size_t j = 0; j < indices.size(); j++) {
            int index = indices[j];
            forward(&mnist.train_images[index][0]);

            double y_truth[10] = {0};
            y_truth[mnist.train_labels[index]] = 1;

            back_prop(y_truth);
            adjust_weights(&mnist.train_images[index][0]);
        }

        std::cout << "Epoch " << i + 1 << " complete. Train accuracy: " << calc_train_accuracy() << "%. Test accuracy: " << calc_test_accuracy() << "%.\n";
    }
}

void NeuralNetwork::forward(double *input_layer) {
    // Matrix multiplication input layer to hidden layer
    mat_mul(a, input_layer, hidden_layer, hidden_layer_size, input_layer_size, 1);

    // Add biases and apply tanh activation
    for (int i = 0; i < hidden_layer_size; i++) {
        hidden_layer[i] += a_bias[i];
        hidden_layer[i] = tanh(hidden_layer[i]);
    }

    // Matrix multiplication hidden layer to output layer
    mat_mul(b, hidden_layer, output_layer, output_layer_size, hidden_layer_size, 1);

    // Add biases and apply sigmoid activation
    for (int i = 0; i < output_layer_size; i++) {
        output_layer[i] += b_bias[i];
        output_layer[i] = 1 / (1 + exp(-output_layer[i]));
    }
}

void NeuralNetwork::back_prop(double *y_truth) {
    // Calculate output layer error
    for (int i = 0; i < output_layer_size; i++) {
        double error = output_layer[i] - y_truth[i];
        output_layer_error[i] = error * output_layer[i] * (1 - output_layer[i]); // derivative of sigmoid
    }

    // Calculate hidden layer error
    for (int i = 0; i < hidden_layer_size; i++) {
        hidden_layer_error[i] = 0;
        for (int j = 0; j < output_layer_size; j++) {
            hidden_layer_error[i] += output_layer_error[j] * b[j * hidden_layer_size + i];
        }
        hidden_layer_error[i] *= (1 - hidden_layer[i] * hidden_layer[i]); // derivative of tanh
    }
}

void NeuralNetwork::adjust_weights(double *inputs) {
    // Adjust weights and biases for hidden layer
    for (int i = 0; i < hidden_layer_size; i++) {
        a_bias[i] -= learning_rate * hidden_layer_error[i];
        for (int j = 0; j < input_layer_size; j++) {
            a[i * input_layer_size + j] -= learning_rate * hidden_layer_error[i] * inputs[j];
        }
    }

    // Adjust weights and biases for output layer
    for (int i = 0; i < output_layer_size; i++) {
        b_bias[i] -= learning_rate * output_layer_error[i];
        for (int j = 0; j < hidden_layer_size; j++) {
            b[i * hidden_layer_size + j] -= learning_rate * output_layer_error[i] * hidden_layer[j];
        }
    }
}

double NeuralNetwork::calc_train_accuracy() {
    int correct = 0;
    for (size_t i = 0; i < mnist.train_images.size(); i++) {
        int predicted = predict(&mnist.train_images[i][0]);
        if (predicted == mnist.train_labels[i]) correct++;
    }
    return 100.0 * correct / mnist.train_images.size();
}

double NeuralNetwork::calc_test_accuracy() {
    int correct = 0;
    for (size_t i = 0; i < mnist.test_images.size(); i++) {
        int predicted = predict(&mnist.test_images[i][0]);
        if (predicted == mnist.test_labels[i]) correct++;
    }
    return 100.0 * correct / mnist.test_images.size();
}

int NeuralNetwork::predict(double *input_layer) {
    forward(input_layer);
    return max_element(output_layer, output_layer + output_layer_size) - output_layer;
}

void NeuralNetwork::init_weights(double *w, int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-0.1, 0.1);

    for (int i = 0; i < rows * cols; i++) {
        w[i] = dist(gen);
    }
}

void NeuralNetwork::init_bias(double *b, int size) {
    fill_n(b, size, 0);
}

void NeuralNetwork::mat_mul(double *a, double *b, double *c, int m, int n, int o) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < o; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k];
            }
            c[i] = sum;
        }
    }
}

void NeuralNetwork::tanh(double *a, int size) {
    for (int i = 0; i < size; i++) {
        double ex = exp(2 * a[i]);
        a[i] = (ex - 1) / (ex + 1);
    }
}

void NeuralNetwork::sigmoid(double *a, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = 1 / (1 + exp(-a[i]));
    }
}

int main()
{
    auto nn = NeuralNetwork();

    auto timer = Timer("Network-CPU");
    nn.learn(1);
    timer.stop();

    return 0;
}
