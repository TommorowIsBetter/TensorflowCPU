#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/5/30 11:01
"""
import numpy as np
import tensorflow as tf
w = tf.Variable(0, dtype=tf.float32)
cost = tf.add(tf.add(w ** 2.0, tf.multiply(-10.0, w)), 25.0)
# cost = w ** 2 - 10 * w + 25
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
for i in range(1000):
    session.run(train)
print(session.run(w))
