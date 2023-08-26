import numpy as np
import random
import data
import matplotlib.pyplot as plt
import time
import pickle

class Network():
    
    def __init__(self, network_size: list) -> None:
        self.size = network_size
        self.num_of_layers = len(network_size)
        self.biases = [np.random.randn(y, 1) for y in network_size[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(network_size[:-1], network_size[1:])]
        
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self, training_data:list[tuple[np.ndarray, np.ndarray]], epochs, mini_batch_size, eta,test_data=None, visualize = False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        
        training_data = list(training_data)

        for current_epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)


            if test_data:
                test_data = list(test_data)
                self.successRate = (self.evaluate(test_data, visualize, current_epoch, epochs) / len(test_data)) * 100
                print(f"epoch: {current_epoch} {self.successRate}")

    def update_mini_batch(self, mini_batch, leariningRate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-((leariningRate/len(mini_batch))*nw) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-((leariningRate/len(mini_batch))*nb) for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, activation, y):
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
            z = np.dot(w, activation)+b
            Z_vectors.append(z)
            activation = sigmoid(z)
            activations.append(sigmoid(z))
        # backward pass
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
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data, visualize, current_epoch, total_epochs):
        if visualize:
            if current_epoch == total_epochs - 1:
                plt.ion()
                fig1, ax1 = plt.subplots()
                array = np.array(test_data[0][0]).reshape(28, 28)
                axim1 = ax1.imshow(array, cmap='gist_gray')
                for testImg, testLabel in test_data:
                    matrix = testImg.reshape(28, 28)
                    axim1.set_data(matrix)
                    fig1.canvas.flush_events()
                    output = self.feedforward(testImg)
                    print(f"prediction: {np.argmax(output)}, conf: {np.max(output) / sum(output)}")

                    time.sleep(0.1)

        test_results = [(np.argmax(self.feedforward(testImg)), np.argmax(vectorizedLabel)) for (testImg, vectorizedLabel) in test_data]

        return sum(int(x == y) for (x, y) in test_results)
    

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return 2*(output_activations-y)
    
    def saveWeightsAndBiases(self, fileName:str):
        with open(f"{fileName} weights", "wb") as weights:
            pickle.dump(self.weights, weights)
        with open(f"{fileName} biases", "wb") as biases:
            pickle.dump(self.biases, biases)
        print("saved")

    def loadWeightsAndBiases(self, fileName:str):
        with open(f"{fileName} weights", "rb") as weights:
            self.weights = pickle.load(weights)
        with open(f"{fileName} biases", "rb") as biases:
            self.biases = pickle.load(biases)
        print("loaded")
    
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



net = Network([784, 100, 10])
trainingData = data.getPrepredData(r"C:\VsCode\python\machineLearning\machine-learning\.MnistDataFiles\train-images.idx3-ubyte", r"C:\VsCode\python\machineLearning\machine-learning\.MnistDataFiles\train-labels.idx1-ubyte")
testData = data.getPrepredData(r"C:\VsCode\python\machineLearning\machine-learning\.MnistDataFiles\t10k-images.idx3-ubyte", r"C:\VsCode\python\machineLearning\machine-learning\.MnistDataFiles\t10k-labels.idx1-ubyte")

net.SGD(trainingData, 30, 10, 3.1, test_data=testData, visualize=True)
if net.successRate > 95:
    net.saveWeightsAndBiases("test")
