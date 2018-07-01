

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


train_size = x_train.shape[0]
batch_size = 10
batck_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
y_batch = t_train[batch_mask]


a = np.array([1,2,3])
a.reshape(1, a.size)

b = np.array([[1,2], [3,4]])
b.size


import numpy as np
import matplotlib.pylab as plt

def function_1 (x):
    return 0.01 * x**2 + 0.1 * x

x = np.arange(0, 20, 1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()




