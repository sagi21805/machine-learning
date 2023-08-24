import numpy as np
import random
import data
import matplotlib.pyplot as plt
import time

class Network():
    
    def __init__(self, network_size: list) -> None:
        self.size = network_size
        self.num_of_layers = len(network_size)
        self.biases = [np.random.uniform(-1, 1, (size, 1)) for size in network_size[1:]]
        self.weights = [np.random.uniform(-1, 1, (k, j)) for k, j in zip(network_size[1:], network_size)]
        self.Success = 0

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                for testImg, testLabel in test_data:
                    output = self.feedforward(testImg)
                    self.evaluate(output, testLabel)
                    print(f"SucessRate: {self.Success / len(test_data)}")
            print(j)

    def update_mini_batch(self, mini_batch, leariningRate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            nabla_b, nabla_w = self.backprop(x, y)
        self.weights = [w-(leariningRate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(leariningRate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, activation:np.ndarray , y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activations = [activation] # list to store all the activations, layer by layer
        Z_vectors = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            Z_vectors.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #********name delta is for convinience but this is the delta of the last layer**************
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(Z_vectors[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())


        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_of_layers):
            z = Z_vectors[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    #TODO improve evaluate this is not working
    def evaluate(self, output, label):
        if np.argmax(output) == np.argmax(label):
            self.Success += 1
            


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return 2*(output_activations-y)
    
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return (-np.exp(-z))/(1+np.exp(-z))**2

net = Network([784, 100, 10])
trainingData = data.getPrepredData(r"C:\VsCode\python\machineLearning\machine-learning\.MnistDataFiles\train-images.idx3-ubyte", r"C:\VsCode\python\machineLearning\machine-learning\.MnistDataFiles\train-labels.idx1-ubyte")
testData = data.getPrepredData(r"C:\VsCode\python\machineLearning\machine-learning\.MnistDataFiles\train-images.idx3-ubyte", r"C:\VsCode\python\machineLearning\machine-learning\.MnistDataFiles\train-labels.idx1-ubyte")

net.SGD(trainingData, 10, 10, 2, test_data=testData)