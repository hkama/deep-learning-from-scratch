

import sys, os
sys.path.append(os.pardir)
sys.path.append('../../')
import numpy as np
from dataset.mnist import load_mnist
from layers import *
import matplotlib.pylab as plt



(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 1000
# iters_num = 100

train_size = x_train.shape[0]
batch_size = 100
# batch_size = 10
learning_rate = 0.1

train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []

iter_per_epoch = max(train_size / batch_size, 1)

fig, ax = plt.subplots(1, 1)
lines, = ax.plot(0, 0)


for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'W2', 'b1', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if True: # i % iter_per_epoch == 0:
        train_accuracy = network.accuracy(x_train, t_train)
        test_accuracy = network.accuracy(x_test, t_test)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        print(train_accuracy, test_accuracy)
        # plt.xlabel("i")
        # plt.ylabel("accuracy")
        # # import pdb; pdb.set_trace()
        # plt.plot(np.arange(i+1), test_accuracy_list)
        # plt.show()


        lines.set_data(np.arange(i+1), test_accuracy_list)
        # set_data()を使うと軸とかは自動設定されないっぽいので，
        # 今回の例だとあっという間にsinカーブが描画範囲からいなくなる．
        # そのためx軸の範囲は適宜修正してやる必要がある．
        ax.set_xlim((0, i))
        ax.set_ylim((0, 1))
        # 一番のポイント
        # - plt.show() ブロッキングされてリアルタイムに描写できない
        # - plt.ion() + plt.draw() グラフウインドウが固まってプログラムが止まるから使えない
        # ----> plt.pause(interval) これを使う!!! 引数はsleep時間
        plt.pause(.01)
        
    
