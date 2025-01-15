import numpy as np
from numba import njit, float64, boolean

def import_from_torch(dataset):
    data_input = []
    data_output = []
    for x, y in dataset:
        data_input.append(x.numpy().squeeze())
        data_output.append(y)
    return np.array(data_input), np.array(data_output)

def class_to_array(maximum_class, x):
    data = np.zeros((maximum_class, 1)) + 0.01
    data[x] = 0.99
    return data

def kfold(k, data, seed=99):
    np.random.seed(seed)
    data = np.random.permutation(data)
    fold_size = len(data) // k
    return data[fold_size*2:], data[:fold_size], data[fold_size:fold_size*2]

@njit(float64[:, :](float64[:, :], boolean))
def sigmoid(x, derivative):
    if derivative:
        return x * (1.0 - x)
    else:
        return 1.0 / (1.0 + np.exp(-x))

@njit(float64[:, :](float64[:, :], boolean))
def relu(x, derivative):
    if derivative:
        return np.where(x <= 0, 0.0, 1.0)
    else:
        return np.maximum(0, x)

@njit(float64[:, :](float64[:, :], boolean))
def leaky_relu(x, derivative):
    if derivative:
        return np.where(x <= 0, 0.01, 1.0)
    else:
        return np.where(x <= 0, 0.01 * x, x)

@njit(float64[:, :](float64[:, :], boolean))
def softmax(x, derivative):
    if derivative:
        pass  # Implement if required for backpropagation
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)
