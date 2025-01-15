import numpy as np
import nn as nn
import utils as h
import time

def load_data():
    inputs = np.load("data/fashion_mnist_train_inputs.npy")
    outputs = np.load("data/fashion_mnist_train_outputs.npy")
    return inputs, outputs

def main():
    train_inputs, train_outputs = load_data()
    
    # network parameters
    layer_sizes = [784, 128, 64, 10]  # Example sizes: input, hidden1, hidden2, output
    activations = [h.sigmoid, h.sigmoid, h.softmax]  

    nn = nn.make_neural_network(layer_sizes, activations, learning_rate=0.1)
    
    start_time = time.time()
    nn.train_network(train_inputs[:1000], train_outputs[:1000], nn, epochs=10)
    print("Training time:", time.time() - start_time)

    prediction = nn.predict(train_inputs[0], nn)
    print("Prediction for first image:", prediction)
    print("Actual output for first image:", train_outputs[0])

if __name__ == "__main__":
    main()
