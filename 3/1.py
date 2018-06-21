

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

print (x_train.shape)
print (t_train.shape)
print (x_test.shape)
print (t_test.shape)

import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.argmax(x, axis = 1)
print(y)




