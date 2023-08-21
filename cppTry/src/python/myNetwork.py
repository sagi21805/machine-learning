import numpy as np

class Network():
    
    def __init__(self, network_size: list) -> None:
        self.size = network_size
        self.num_of_layers = len(network_size)
        self.biases = [np.random.randn(size, 1) for size in network_size[1:]]
        self.weights = [np.random.randn(k, j) for k, j in zip(network_size[1:], network_size)]

    def feedforward(self, layer: np.ndarray):
        #gets the input neuron values of the network and returns the output values
        for layerWeights, layerBiases in zip(self.weights, self.biases):
            print(f"w: {layerWeights}")
            print(f"b: {layerBiases}")
        return 
    
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return (-np.exp(-z))/(1+np.exp(-z))**2

net = Network([3, 4, 2, 1])
net.feedforward(2)