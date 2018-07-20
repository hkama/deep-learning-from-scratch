#  coding: utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.gradient import numerical_gradient
from collections import OrderedDict

class ReLu:
    def __init__(self):
        self.mask == None
    def forward(self, x):
        self.mask = x<=0
        out = x.copy
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dx = dout.copy
        dx[self.mask] = 0
        return dx
        
class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
    def backward(self, dout):
        return dout * self.out * (1 - self.out)
        
class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.dw = None
        self.db = None
        dx = None
    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b
    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        pass
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        loss = cross_entropy_error(x, t)
        return loss
    def backward(self, dout=1):
        # しっかり間違えた
        # dx = self.y - self.t ではない
        # cross entropy error 計算でbatch_sizeで割られているため、影響量はbatch_size分小さくなるはず
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

        
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = weight_init_std * np.random.randn(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = weight_init_std * np.random.randn(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLu'] = ReLu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastlayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if not t.ndim == 1:
            t = np.argmax(t, axis=1)
        # return np.sum(y==t) / t.shape[0]
        # floatをつけなかった
        accuracy = np.sum(y==t) / float(t.shape[0])
        return accuracy
    def numerical_gradient(self, x, t):
        loss_w = lambda W: self.loss(x,t)
        grads = {}
        grads{'W1'} = numerical_gradient(loss_w, self.params['W1'])
        grads{'W2'} = numerical_gradient(loss_w, self.params['W2'])
        grads{'b1'} = numerical_gradient(loss_w, self.params['b1'])
        grads{'b2'} = numerical_gradient(loss_w, self.params['b2'])
        return grads
    def gradient(self):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastlayer.backward(dout)
        layers = self.layers.values():
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads={}
        grads{'W1'} = self.layers.['Affine1'].dw
        grads{'b1'} = self.layers.['Affine1'].db
        grads{'W2'} = self.layers.['Affine2'].dw
        grads{'b2'} = self.layers.['Affine2'].db
        return grads
        
