import numpy as np
from numba import jit
class Network():
    
    def __init__(self, network_size: list) -> None:
        self.size = network_size
        self.num_of_layers = len(network_size)
        self.biases = [np.random.randn(size, 1) for size in network_size[1:]]
        self.weights = [np.random.randn(k, j) for k, j in zip(network_size[1:], network_size)]

    def feedForward(self, firstLayer):
        currentLayer = np.array(firstLayer)
        for index in range(len(self.size)-1):
            nextLayer = np.zeros((self.size[index+1], 1))
            for layerIndex in range(len(self.weights[index])):
                nextLayer[layerIndex] = sigmoid(np.sum(currentLayer * self.weights[index][layerIndex])) + self.biases[index][layerIndex]
            currentLayer = nextLayer
        
        return currentLayer

def sigmoid(z):
    return 1/(1 + np.exp(-z))

net = Network([784, 100, 10])
print(net.feedForward(np.random.rand(784, 1)))
