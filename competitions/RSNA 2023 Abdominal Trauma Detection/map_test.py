
import tensorflow as tf
import numpy as np
from cv2 import imread
def add(a, b):
    return a + b

number1 = ['1', '2', '3', '4']
def p(a, b):
    for x in a: 
        print(x)
    return a


d = tf.data.Dataset.from_tensor_slices((number1, number1)).map(add)
for x in d:
    print(x)

