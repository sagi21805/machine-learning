import time
import data
import matplotlib.pyplot as plt
import numpy as np

test_data = data.getPrepredData(r"C:\VsCode\python\machineLearning\machine-learning\.MnistDataFiles\t10k-images.idx3-ubyte", r"C:\VsCode\python\machineLearning\machine-learning\.MnistDataFiles\t10k-labels.idx1-ubyte")


plt.ion()
fig1, ax1 = plt.subplots()
array = np.array(test_data[0][0]).reshape(28, 28)
axim1 = ax1.imshow(array, cmap='gist_gray')

for testImg, testLabel in test_data:
    matrix = testImg.reshape(28, 28)
    axim1.set_data(matrix)
    fig1.canvas.flush_events()
    print(np.argmax(testLabel))
    time.sleep(0.7)