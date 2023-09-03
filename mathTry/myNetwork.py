import numpy as np
import random
import data
import matplotlib.pyplot as plt
import time
import pickle
from cost_functions import Cost, CrossEntropyCost, QuadraticCost
import matplotlib
from numba import njit

matplotlib.use("TkAgg")

class Network():
    
    def __init__(self, network_size: list, cost = CrossEntropyCost) -> None:
        self.size = network_size
        self.num_of_layers = len(network_size)
        self.biases = [np.random.randn(y, 1) for y in network_size[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(network_size[:-1], network_size[1:])]
        self.cost = cost
    
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self, training_data:list[tuple[np.ndarray, np.ndarray]], epochs, mini_batch_size, learningRate ,test_data=None, visualize = False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` where ``x`` is the input and ``y`` is the expected output
        The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        the ``visualize`` shows the input numbers and prints the networks guess with the confidence level on the last epoch"""
        
        training_data = list(training_data)

        for current_epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learningRate)


            if test_data:
                test_data = list(test_data)
                self.successRate = (self.evaluate(test_data, visualize, current_epoch, epochs) / len(test_data)) * 100
                print(f"epoch: {current_epoch} {self.successRate}")

    def update_mini_batch(self, mini_batch, leariningRate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)`` where ``x`` is the input
        and ``y`` is the expected output"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            nabla_b, nabla_w = backprop(self.weights, self.biases, CrossEntropyCost, self.num_of_layers, x, y)
            # nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            self.weights = [w-((leariningRate/len(mini_batch))*nw) for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-((leariningRate/len(mini_batch))*nb) for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data, visualize, current_epoch, total_epochs):
        """test the network against the test data and print the success rate on eache epoch

        Args:
            ``test_data`` (list[tuple[input, wanted output]]): the test data the networks is tested with
        ``visualize`` (boolean):shows the input numbers and prints the networks guess with the confidence level on the last epoch
            ``current_epoch`` (int): the current epoch the network is training on
        ``total_epochs`` (int): the total epochs the networks trains on

        Returns:
            int: the total images the network succeeded on
        """
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
                    print(f"prediction: {np.argmax(output)}, True: {np.argmax(testLabel)}, conf: {np.max(output) / sum(output)}")
                    plt.show()
                    time.sleep(0.5)

        test_results = [(np.argmax(self.feedforward(testImg)), np.argmax(vectorizedLabel)) for (testImg, vectorizedLabel) in test_data]

        return sum(int(x == y) for (x, y) in test_results)
    
    def saveWeightsAndBiases(self, file_name:str):
        with open(f"W{file_name}", "wb") as weights:
            pickle.dump(self.weights, weights)
        with open(f"B{file_name}", "wb") as biases:
            pickle.dump(self.biases, biases)
        print("saved")

    def loadWeightsAndBiases(self, weights_path:str, bias_path: str):
        with open(f"{weights_path}", "rb") as weights:
            self.weights = pickle.load(weights)
        with open(f"{bias_path}", "rb") as biases:
            self.biases = pickle.load(biases)
        print("loaded")
    
@njit(fastmath=True)
def sigmoid(z):
    """the sigmoif function"""
    return 1/(1 + np.exp(-z))

@njit(fastmath=True)
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

 
# @njit(fastmath=True)
def backprop(weights_mat: np.ndarray, bias_mat: np.ndarray, cost: Cost, num_of_layers: int, activation, y):
    """Return a tuple ``(nabla_b, nabla_w)`` representing the
    gradient for the cost function C_x.  ``nabla_b`` and
    ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    to ``self.biases`` and ``self.weights``."""
    nabla_b = [np.zeros(b.shape) for b in bias_mat]
    nabla_w = [np.zeros(w.shape) for w in weights_mat]
    # feedforward
    activations = [activation] # list to store all the activations, layer by layer
    Z_vectors = [] # list to store all the z vectors, layer by layer
    for b, w in zip(bias_mat, weights_mat):
        z = np.dot(w, activation) + b
        Z_vectors.append(z)
        activation = sigmoid(z)
        activations.append(activation)
        
    # backward pass
    delta = (cost).delta(Z_vectors[-1], activations[-1], y)
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    
    #this loop is going through the network backwords to calulate the error
    
    for l in range(2, num_of_layers):
        z = Z_vectors[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(weights_mat[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
    return (nabla_b, nabla_w)







net = Network([784, 100, 10])
trainingData = data.getPrepredData(r"/home/sagi21805/Desktop/Vscode/machine-learning/.MnistDataFiles/train-images.idx3-ubyte", r"/home/sagi21805/Desktop/Vscode/machine-learning/.MnistDataFiles/train-labels.idx1-ubyte")
testData = data.getPrepredData(r"/home/sagi21805/Desktop/Vscode/machine-learning/.MnistDataFiles/t10k-images.idx3-ubyte", r"/home/sagi21805/Desktop/Vscode/machine-learning/.MnistDataFiles/t10k-labels.idx1-ubyte")

start_time = time.time()

net.SGD(trainingData, 30, 10, 0.15, test_data = testData, visualize=False)
if net.successRate > 95:
    net.saveWeightsAndBiases(f"{net.size}-{net.successRate}.pickle")

print(f"time: {time.time() - start_time}")

# net.loadWeightsAndBiases(r"C:\VsCode\python\machineLearning\machine-learning\mathTry\W[784, 16, 16, 10]-93.56.pickle", r"C:\VsCode\python\machineLearning\machine-learning\mathTry\B[784, 16, 16, 10]-93.56.pickle")
# img = cv2.cvtColor(cv2.imread(r"C:\VsCode\python\machineLearning\machine-learning\mathTry\testImages\test.png"), cv2.COLOR_BGR2GRAY) / 255
# output = net.feedforward(np.reshape(img, (784, 1)))
# print(np.argmax(output))
# print(f"{np.max(output) / sum(output)}")
# plt.imshow(img, cmap='gist_gray')
# plt.show()



