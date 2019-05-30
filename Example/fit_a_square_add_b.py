#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/5/30 14:35
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def add_layer(inputs, insize, outsize, activation_func=None):
  Weights = tf.Variable(tf.random_normal([insize, outsize]))
  bias = tf.Variable(tf.zeros([1, outsize])+0.1)
  wx_plus_b = tf.matmul(inputs,Weights) + bias
  if activation_func:
      return activation_func(wx_plus_b)
  else:
      return wx_plus_b


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


l1 = add_layer(xs, 1, 10, activation_func=tf.nn.relu)
preds = add_layer(l1, 10, 1, activation_func=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - preds), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(2000):
      sess.run(train, feed_dict={xs: x_data, ys: y_data})
      if i % 50 == 0:
          preds_val = sess.run(preds, feed_dict={xs: x_data, ys: y_data})
          try:
              ax.lines.remove(lines[0])
          except:
              pass
          lines = ax.plot(x_data, preds_val, 'r-', lw=5)
          plt.pause(0.5)
