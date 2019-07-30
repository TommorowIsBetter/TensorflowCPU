#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/7/30 15:10
"""
import tensorflow as tf
# 定义一个常量，然后经过relu激活函数，查看具体的值
out = tf.constant([[-1, -2, 3]], dtype=tf.float32, name='out')
re_ = tf.nn.relu(out)

# 初始化随机变量，然后查看具体的值
Weights = tf.Variable(tf.random_normal([2, 3]))
biases = tf.Variable(tf.zeros([1, 10]) + 0.1)
with tf.Session() as sess:
    print(sess.run(re_))
    sess.run(tf.global_variables_initializer())
    print(sess.run(Weights))
    print(sess.run(biases))

