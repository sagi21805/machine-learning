import numpy as np

network_size = [2, 1]

a = np.array([2, 3])
w = [np.random.randn(k, j) for k, j in zip(network_size[1:], network_size)]



def feedForward(layer, AllweightList):
    sum = np.zeros((a.shape))
    for weights in AllweightList:
        for index, num in enumerate(layer):
            sum += num*weights[index] 

feedForward(a, w)