#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/6/16 21:05
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
batchsize = 64
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def train(init_bias_):
    sess = tf.InteractiveSession()
    # ---------------------------------------初始化网络结构-------------------------------------
    x = tf.placeholder("float", [None, 784], name='x-input')
    y_ = tf.placeholder("float", [None, 10], name='y-input')
    W1 = tf.Variable(tf.random_uniform([784, 100], -0.5 + init_bias_, 0.5 + init_bias_))
    b1 = tf.Variable(tf.random_uniform([100], -0.5 + init_bias_, 0.5 + init_bias_))
    u1 = tf.matmul(x, W1) + b1
    y1 = tf.nn.sigmoid(u1)
    W2 = tf.Variable(tf.random_uniform([100, 10], -0.5 + init_bias_, 0.5 + init_bias_))
    b2 = tf.Variable(tf.random_uniform([10], -0.5 + init_bias_, 0.5 + init_bias_))
    y = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)
    # ---------------------------------------设置网络的训练方式-------------------------------------
    mse = tf.reduce_sum(tf.square(y - y_))
    train_step = tf.train.AdamOptimizer(0.001).minimize(mse)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    init = tf.global_variables_initializer()
    sess.run(init)
    # ---------------------------------------开始训练-------------------------------------
    for i_ in range(5):
        batch_xs, batch_ys = mnist.train.next_batch(batchsize)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print('权重初始化范围[%.1f,%.1f],1000次训练过后的准确率' % (init_bias_ - 0.5, init_bias_ + 0.5),
          sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    # 权重的初始化时的偏置量
    init_bias = -0.6
    for i in range(11):
        init_bias += 0.1
        train(init_bias)
