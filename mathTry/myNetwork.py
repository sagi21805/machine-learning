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
    
    def __init__(self, network_size: list, cost: Cost = CrossEntropyCost) -> None:
        self.size = network_size
        self.num_of_layers = len(network_size)
        self.cost = cost
    
    
    def initialize_random_weights_biases(self):
        """initializes random ``weights`` and ``biases`` for the net"""
        self.biases = [np.random.randn(y, 1) for y in self.size[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.size[:-1], self.size[1:])]
    
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def train(self, training_data:list[tuple[np.ndarray, np.ndarray]], epochs: int, mini_batch_size: int, learning_rate:float ,test_data: list[tuple[np.ndarray, np.ndarray]] = None, visualize = False):
        """applys ``Stochstic gradient decent`` with the ``backpropagetion`` algorithem to train the net\n

        Args:
            ``training_data`` list of tuples with (``input``:np.ndarray, ``expected output``: np.ndarray)\n
            ``epochs`` (int): number of times going through the training data\n
            ``mini_batch_size`` (int): size of each mini batch\n
            ``learning_rate (float)``: constant number that represents the learning rate\n
            ``test_data (optional)``:same as the training data, but for testing (Defaults to None)\n
            ``visualize (optional)``: visualize the testing of the last epoch (Defaults to False)\n
        """
        
        training_data = list(training_data)

        for current_epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)


            if test_data:
                test_data = list(test_data)
                self.successRate = (self.evaluate(test_data, visualize, current_epoch, epochs) / len(test_data)) * 100
                print(f"epoch: {current_epoch} {self.successRate}")

    def update_mini_batch(self, mini_batch, leariningRate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch"""
        
        for x, y in mini_batch:
            nabla_b, nabla_w = self.backprop(x, y)
            self.weights = [w-((leariningRate/len(mini_batch))*nw) for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-((leariningRate/len(mini_batch))*nb) for b, nb in zip(self.biases, nabla_b)]
            
            
    def backprop(self, activation: np.ndarray, y: np.ndarray):
        """``the backpropagation algorithem``

        Args:
            ``activation``: the input to the net\n
            ``y`` : the wanted outout

        Returns:
            ``tuple(nabla_b, nabla_w)`` the changes of the weights and biases (the same shape as the weights and biases)
        """
        
        
        if self.weights and self.biases:
            
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            activations = [activation] # list to store all the activations, layer by layer
            Z_vectors = [] # list to store all the z vectors, layer by layer
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w, activation) + b
                Z_vectors.append(z)
                activation = sigmoid(z)
                activations.append(activation)
                
            # backward pass
            delta = (self.cost).delta(Z_vectors[-1], activations[-1], y)
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            
            #this loop is going through the network backwords to calulate the error
            
            for l in range(2, self.num_of_layers):
                z = Z_vectors[-l]
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
            return (nabla_b, nabla_w)
    
    def evaluate(self, test_data, visualize, current_epoch, total_epochs):
        """test the network against the test data and print the success rate on eache epoch

        Args:
            ``test_data`` the test data the networks is tested with\n
            ``visualize`  shows the input numbers and prints the networks guess with the confidence level on the last epoch\n
            ``current_epoch`` the current epoch the network is training on\n
            ``total_epochs``  the total epochs the networks trains on\n

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
    
    def save_weights_and_biases(self, file_name:str):
        """``saves the weights and biases of the network to a binary file with pickle``

        Args:
            ``file_name`` the name of the file of the weights and biases
        """
        
        with open(f"W{file_name}", "wb") as weights:
            pickle.dump(self.weights, weights)
            
        with open(f"B{file_name}", "wb") as biases:
            pickle.dump(self.biases, biases)
            
        print("saved")

    def load_weights_and_biases(self, weights_path:str, bias_path: str):
        """``load the weights and biases with pickle``

        Args:
            ``weights_path`` the path to the weights file\n
            ``bias_path`` the path to the biases file
        """
        with open(f"{weights_path}", "rb") as weights:
            self.weights = pickle.load(weights)
            
        with open(f"{bias_path}", "rb") as biases:
            self.biases = pickle.load(biases)
            
        print("loaded")
    
@njit(fastmath=True)
def sigmoid(z):
    """``the sigmoid function``"""
    return 1/(1 + np.exp(-z))

@njit(fastmath=True)
def sigmoid_prime(z):
    """`Derivative of the sigmoid function`"""
    return sigmoid(z)*(1-sigmoid(z))


net = Network([784, 100, 10])
trainingData = data.getPrepredData(r"/home/sagi21805/Desktop/Vscode/machine-learning/.MnistDataFiles/train-images.idx3-ubyte", r"/home/sagi21805/Desktop/Vscode/machine-learning/.MnistDataFiles/train-labels.idx1-ubyte")
testData = data.getPrepredData(r"/home/sagi21805/Desktop/Vscode/machine-learning/.MnistDataFiles/t10k-images.idx3-ubyte", r"/home/sagi21805/Desktop/Vscode/machine-learning/.MnistDataFiles/t10k-labels.idx1-ubyte")

start_time = time.time()

net.initialize_random_weights_biases()
net.train(trainingData, 1, 10, 0.15, test_data = testData, visualize=False)
if net.successRate > 95:
    net.save_weights_and_biases(f"{net.size}-{net.successRate}.pickle")

print(f"time: {time.time() - start_time}")

# net.loadWeightsAndBiases(r"C:\VsCode\python\machineLearning\machine-learning\mathTry\W[784, 16, 16, 10]-93.56.pickle", r"C:\VsCode\python\machineLearning\machine-learning\mathTry\B[784, 16, 16, 10]-93.56.pickle")
# img = cv2.cvtColor(cv2.imread(r"C:\VsCode\python\machineLearning\machine-learning\mathTry\testImages\test.png"), cv2.COLOR_BGR2GRAY) / 255
# output = net.feedforward(np.reshape(img, (784, 1)))
# print(np.argmax(output))
# print(f"{np.max(output) / sum(output)}")
# plt.imshow(img, cmap='gist_gray')
# plt.show()



