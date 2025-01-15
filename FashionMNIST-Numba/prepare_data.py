import numpy as np
import utils as h
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def prepare_fashion_mnist_data():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    batch_size = 64 
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    train_inputs, train_outputs = [], []
    for x, y in train_dataloader:
        x = x.numpy()  
        y = y.numpy()
        train_inputs.append(x.reshape(x.shape[0], -1))
        train_outputs.extend([h.class_to_array(10, label) for label in y])

    train_inputs = np.vstack(train_inputs)  
    train_outputs = np.vstack(train_outputs)

    np.save("data/fashion_mnist_train_inputs", train_inputs)
    np.save("data/fashion_mnist_train_outputs", train_outputs)

if __name__ == "__main__":
    prepare_fashion_mnist_data()
