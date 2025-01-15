import numpy as np
from numba import njit, types, typed, prange
from numba.experimental import jitclass

spec = [
    ('layer_sizes', types.ListType(types.int64)),
    ('layer_activations', types.ListType(types.FunctionType(types.float64[:, ::1](types.float64[:, ::1], types.boolean)))),
    ('weights', types.ListType(types.float64[:, ::1])),
    ('biases', types.ListType(types.float64[:, ::1])),
    ('layer_outputs', types.ListType(types.float64[:, ::1])),
    ('learning_rate', types.float64),
]

@jitclass(spec)
class NeuralNetwork:
    def __init__(self, layer_sizes, layer_activations, weights, biases, layer_outputs, learning_rate):
        self.layer_sizes = layer_sizes
        self.layer_activations = layer_activations
        self.weights = weights
        self.biases = biases
        self.layer_outputs = layer_outputs
        self.learning_rate = learning_rate

@njit
def make_neural_network(layer_sizes, activation_funcs, learning_rate=0.1, low=-0.5, high=0.5):
    typed_layer_sizes = typed.List.empty_list(types.int64)
    for size in layer_sizes:
        typed_layer_sizes.append(size)

    typed_layer_activations = typed.List.empty_list(types.FunctionType(types.float64[:, ::1](types.float64[:, ::1], types.boolean)))
    for func in activation_funcs:
        typed_layer_activations.append(func)

    typed_weights = typed.List.empty_list(types.float64[:, ::1])
    typed_biases = typed.List.empty_list(types.float64[:, ::1])
    typed_layer_outputs = typed.List.empty_list(types.float64[:, ::1])

    for i in range(len(layer_sizes) - 1):
        weights = np.random.uniform(low, high, (layer_sizes[i+1], layer_sizes[i]))
        biases = np.random.uniform(low, high, (layer_sizes[i+1], 1))
        typed_weights.append(weights)
        typed_biases.append(biases)
        typed_layer_outputs.append(np.zeros((layer_sizes[i+1], 1)))

    typed_layer_outputs.append(np.zeros((layer_sizes[-1], 1)))
    return NeuralNetwork(typed_layer_sizes, typed_layer_activations, typed_weights, typed_biases, typed_layer_outputs, learning_rate)

@njit
def feed_forward(input_data, nn):
    current_activation = input_data
    nn.layer_outputs[0] = current_activation
    
    for i in range(len(nn.weights)):
        current_activation = nn.layer_activations[i](np.dot(nn.weights[i], current_activation) + nn.biases[i], False)
        nn.layer_outputs[i+1] = current_activation
    
    return current_activation

@njit
def back_propagation(target, nn):
    errors = target - nn.layer_outputs[-1]
    for i in range(len(nn.weights)-1, -1, -1):
        delta = errors * nn.layer_activations[i](nn.layer_outputs[i+1], True)
        errors = np.dot(nn.weights[i].T, delta)
        gradient = np.dot(delta, nn.layer_outputs[i].T)
        
        nn.weights[i] += nn.learning_rate * gradient
        nn.biases[i] += nn.learning_rate * delta

@njit
def train_network(train_inputs, train_outputs, nn, epochs):
    for epoch in range(epochs):
        for x, y in zip(train_inputs, train_outputs):
            feed_forward(x.reshape(-1, 1), nn)
            back_propagation(y.reshape(-1, 1), nn)

@njit
def predict(input_data, nn):
    return feed_forward(input_data.reshape(-1, 1), nn)
