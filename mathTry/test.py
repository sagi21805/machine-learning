import numpy as np
sizes = [4, 3]
a = np.array([1, 2, 3, 4])
a = a.reshape(4, 1)
b = [np.random.uniform(-1, 1, (y, 1)) for y in sizes[1:]]
w = [np.random.uniform(-1, 1, (y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
print(w[0], "\n\n")
a = np.dot(w[0], a)

print(a)
